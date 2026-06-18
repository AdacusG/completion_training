#!/usr/bin/env python3
import os
import pickle
import torch
import torch.nn as nn
import sys

# --- 1. Setup & Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'
out_dir = 'out_1char'

# These must match your training hyperparameters exactly!
embedding_dim = 128      # Size of your token vectors
n_heads = 4
n_layers = 4
seq_len = 8  # Set this to match your training max_len (2n + 1)

# --- 2. Load Metadata Vocabulary ---
data_dir = '1-Char/data'
with open(os.path.join(data_dir, 'meta.pkl'), 'rb') as f:
    meta = pickle.load(f)
vocab_size = meta['vocab_size']
stoi = meta['stoi']
itos = meta['itos']

EOT_ID = stoi['\n']
PAD_ID = stoi['_']

# --- 3. Rebuild the Model Architecture ---
class PureCompletionTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(seq_len, d_model)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=0.0, batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        b, t = idx.size()
        pos = torch.arange(0, t, dtype=torch.long, device=idx.device).unsqueeze(0)
        x = self.token_embedding(idx) + self.position_embedding(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(t, device=idx.device)
        x = self.transformer(x, memory=x, tgt_mask=mask, memory_mask=mask)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

# --- 4. Load the Saved Weights ---
model = PureCompletionTransformer(vocab_size, embedding_dim, n_heads, n_layers).to(device)
weights_path = os.path.join(out_dir, 'completion_model.pth')

if not os.path.exists(weights_path):
    raise FileNotFoundError(f"Could not find model weights at {weights_path}")

model.load_state_dict(torch.load(weights_path, map_location=device))
model.eval()
print("Model weights successfully loaded! Entering generation loop...\n")

# --- 5. Autoregressive Completion Function ---
def complete_sequence(prompt_str):
    clean_prompt = prompt_str.replace("=", "")
    
    # Max input/output string length 'n' is calculated from total chars minus syntax tokens (= and \n) divided by 2
    n = (seq_len - 2) // 2  # (8 - 2) // 2 = 3
    needed_padding = n - len(clean_prompt)
    
    # Left-pad to perfectly match your input.txt data structure rule
    padded_prompt = ("_" * needed_padding) + clean_prompt + "="
    
    # Tokenize characters into integers
    tokens = [stoi[c] for c in padded_prompt]
    x = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    
    # Loop to predict one character at a time autoregressively
    with torch.no_grad():
        while x.size(1) < seq_len:
            logits = model(x)
            next_token_logits = logits[0, -1, :]
            next_token_id = torch.argmax(next_token_logits).item()
            
            x = torch.cat((x, torch.tensor([[next_token_id]], device=device)), dim=1)
            
            # Stop if the model cleanly predicts the newline
            if next_token_id == EOT_ID:
                break
                
    completed_chars = [itos[t.item()] for t in x[0]]
    return "".join(completed_chars)

# --- 6. Test Prompt Execution ---
test_prompt = sys.argv[1] if len(sys.argv) > 1 else "a=" 
output = complete_sequence(test_prompt)
print(f"Input Prompt : {test_prompt}")
print(f"Model Output : {output}")