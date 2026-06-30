#!/usr/bin/env python3
import os
import pickle
import torch
import torch.nn as nn

# --- 1. Setup & Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = 'out_1char' 

# These must match your training hyperparameters exactly!
embedding_dim = 128      
n_heads = 4
n_layers = 4

# Load the weights first to see how large the positional embedding matrix actually is
weights_path = os.path.join(out_dir, 'completion_model.pth')
if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Could not find model weights at {weights_path}")
checkpoint = torch.load(weights_path, map_location=device)

# Dynamically extract the maximum sequence length the model supports
max_supported_len = checkpoint['position_embedding.weight'].shape[0]

# --- 2. Load Metadata Vocabulary ---
data_dir = '1-Char/data'
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']

EOT_ID = stoi['\n']
PAD_ID = stoi['_']

# --- 3. Rebuild the Model Architecture (MATCHES TRAINING) ---
class PureCompletionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_supported_len, d_model)
        
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
        x = self.token_embedding(idx) + self.position_embedding(pos)
        
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# --- 4. Load the Saved Weights ---
model = PureCompletionTransformer(vocab_size, embedding_dim, n_heads, n_layers).to(device)
model.load_state_dict(checkpoint)
model.eval()
print("Model weights successfully loaded! Running systematic evaluation...\n")

# --- 5. Autoregressive Completion Function ---
def complete_sequence(prompt_str, max_new_tokens=5):
    if not prompt_str.endswith("="):
        prompt_str += "="
            
    # Quick vocabulary check to prevent crashes if characters are missing
    for c in prompt_str:
        if c not in stoi:
            return f"Error: Char '{c}' not in vocab"

    tokens = [stoi[c] for c in prompt_str]
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    start_len = x.size(1)
    
    with torch.no_grad():
        while (x.size(1) - start_len) < max_new_tokens and x.size(1) < max_supported_len:
            logits = model(x)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            new_token_tensor = torch.tensor([[next_token_id]], device=device)
            x = torch.cat((x, new_token_tensor), dim=1)
            
            if next_token_id == EOT_ID:
                break
                
    completed_chars = [itos[t.item()] for t in x[0]]
    
    if "=" in completed_chars:
        eq_idx = completed_chars.index("=")
        rhs_chars = completed_chars[eq_idx + 1:]
        return "".join(rhs_chars).replace("\n", "").replace("_", "")
    else:
        return "".join(completed_chars)

# --- 6. Exhaustive Combinatorial Evaluation Loop ---
correct_predictions = 0
total_predictions = 0

print("Evaluating all pairs from 0+0 to 299+299...")

for i in range(0, 300):
    for j in range(0, 300):
        # Programmatically construct prompt and the correct math answer
        prompt = f"{i}+{j}="
        ground_truth = str((i + j) % 300)
        
        # Request enough generated tokens to fit the answer string length safely
        prediction = complete_sequence(prompt, max_new_tokens=len(ground_truth) + 1)
        
        if prediction == ground_truth:
            correct_predictions += 1
            
        total_predictions += 1

# --- 7. Metrics Report Summary ---
print("\n" + "="*40)
print("📊 EXHAUSTIVE ACCURACY REPORT (generate_all.py)")
print(f"Total Combinations Evaluated : {total_predictions}")
print(f"Total Correct Math Matches   : {correct_predictions}")
accuracy = (correct_predictions / total_predictions) * 100
print(f"Overall Model Accuracy       : {accuracy:.2f}%")
print("="*40)