#!/usr/bin/env python3
import os
import pickle
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# --- 1. Configuration & Hyperparameters ---
device = 'cpu'
batch_size = 32          # Parallel batching capacity
embedding_dim = 128      # Size of your token vectors
n_heads = 4              # Attention heads
n_layers = 4             # Transformer layers
epochs = 100              # Number of full passes through the data
lr = 1e-3
out_dir = 'out_1char'

os.makedirs(out_dir, exist_ok=True)
torch.manual_seed(42)

# --- 2. Load Metadata Vocabulary ---
data_dir = os.path.join('1-Char/data')
meta_path = os.path.join(data_dir, 'meta.pkl')
if not os.path.exists(meta_path):
    raise FileNotFoundError("Missing meta.pkl! Run your prepare.py script first.")

with open(meta_path, 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']

# Target our special formatting characters from the vocabulary
PAD_ID = stoi['_']    # Your padding token
EQUAL_ID = stoi['=']  # Delimiter splitting inputs and outputs
EOT_ID = stoi['\n']   # End of Text marker

# --- 3. Custom PyTorch Sequence Dataset ---
class SequenceDataset(Dataset):
    def __init__(self, bin_path):
        # Load the raw uint16 token IDs from disk
        raw_tokens = np.fromfile(bin_path, dtype=np.uint16).astype(np.int64)
        
        # Locate exactly where each line terminates (\n)
        eot_indices = np.where(raw_tokens == EOT_ID)[0]
        
        self.sequences = []
        start_idx = 0
        for end_idx in eot_indices:
            # Slice out the exact line sequence (including the \n)
            seq = raw_tokens[start_idx : end_idx + 1]
            self.sequences.append(torch.tensor(seq))
            start_idx = end_idx + 1
            
        # Determine your sequence length automatically from the data geometry (2n + 2)
        self.max_len = max(len(s) for s in self.sequences)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        
        # X is the raw sequence except the last token (which is the \n)
        x = seq[:-1].clone()
        
        # Y is shifted by 1 step forward (Causal language modeling targets)
        # Sequence: [_, a, =, b, \n] -> Targets: [a, =, b, \n]
        y = torch.full_like(x, -100) # Initialize targets to -100 (ignore index)
        shifted_targets = seq[1:].clone()
        
        # Find where the '=' delimiter token lives in this specific line
        # Find where the '=' delimiter token lives in this specific line
        eq_positions = (x == EQUAL_ID).nonzero(as_tuple=True)[0]
        if len(eq_positions) > 0:
            eq_pos = eq_positions[0].item()
            
            # The character following x[eq_pos] is learned from shifted_targets[eq_pos].
            y[eq_pos:] = shifted_targets[eq_pos:]
        else:
            raise ValueError(f"Line {idx} is missing the '=' delimiter!")
            
        # The maximum possible length for our truncated tensors is self.max_len - 1
        target_len = self.max_len - 1
        
        if len(x) < target_len:
            padding_needed = target_len - len(x)
            
            # Pad X with PAD_ID (trailing tokens)
            padding_x = torch.full((padding_needed,), PAD_ID, dtype=torch.long)
            x = torch.cat([x, padding_x])
            
            # Pad Y with -100 so the loss function completely ignores the padded zone
            padding_y = torch.full((padding_needed,), -100, dtype=torch.long)
            y = torch.cat([y, padding_y])
            
        return x, y

# Instantiate our genuine parallel DataLoaders
train_dataset = SequenceDataset(os.path.join(data_dir, 'train.bin'))
val_dataset = SequenceDataset(os.path.join(data_dir, 'val.bin'))

# Set seq_len to match our data dimension natively
seq_len = train_dataset.max_len

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Dataset Loaded! Sequence Vector Width: {seq_len} tokens.")
print(f"Parallel Batch Layout Matrix: ({batch_size}, {seq_len})")

# --- 4. Pure PyTorch Transformer Encoder Architecture ---
class PureCompletionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        
        # Use EncoderLayer for a GPT-style decoder-only behavior
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        
        # Add spatial/positional info to character embeddings
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        # Generate standard causal mask to prevent looking ahead into the answer
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        
        # Run standard self-attention using the causal mask
        x = self.transformer(x, mask=mask, is_causal=True)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# --- 5. Model Initialization & Optimization Configuration ---
model = PureCompletionTransformer(vocab_size, embedding_dim, n_heads, n_layers).to(device)
optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-1)
criterion = nn.CrossEntropyLoss(ignore_index=-100) # Automatically skips input token evaluation!

# --- 6. The Clean Training Loop ---
print(f"Firing up pure PyTorch loop on device: {device}...")
for epoch in range(epochs):
    model.train()
    total_train_loss = 0
    
    for batch_idx, (X, Y) in enumerate(train_loader):
        X, Y = X.to(device), Y.to(device)
        
        # Forward pass: process all rows simultaneously
        logits = model(X)
        
        # Flatten tensors out to measure cross-entropy dimensions cleanly
        loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
        
        # Backward optimization step
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_train_loss += loss.item()
        
        if batch_idx % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs} | Batch {batch_idx}/{len(train_loader)} | Loss: {loss.item():.4f}")

    # Validation cycle at epoch boundary
    model.eval()
    total_val_loss = 0
    with torch.no_grad():
        for X, Y in val_loader:
            X, Y = X.to(device), Y.to(device)
            logits = model(X)
            val_loss = criterion(logits.view(-1, vocab_size), Y.view(-1))
            total_val_loss += val_loss.item()
            
    print("\n" + "="*60)
    print(f"📊 EPOCH {epoch+1} SUMMARY")
    print(f"Average Train Loss : {total_train_loss/len(train_loader):.4f}")
    print(f"Average Val Loss   : {total_val_loss/len(val_loader):.4f}")
    print("="*60 + "\n")

# Save final clean weights
torch.save(model.state_dict(), os.path.join(out_dir, 'completion_model.pth'))
print("Weights saved successfully to disk. Training pipeline complete!")