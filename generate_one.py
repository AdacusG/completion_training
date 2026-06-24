#!/usr/bin/env python3
import os
import pickle
import torch
import torch.nn as nn
import argparse

def main():
    # --- 1. Setup Argparse ---
    parser = argparse.ArgumentParser(
        description="Run inference on a single prompt using the sequence completion transformer."
    )
    
    # Required positional argument for the text prompt
    parser.add_argument(
        "prompt",
        type=str,
        help="The input prompt string for the model (e.g., 'abc' or 'abc=')"
    )
    
    # Optional flags for directories
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out_1char",
        help="Directory containing the model checkpoint (default: out_1char)"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="1-Char/data",
        help="Directory containing metadata vocabulary (default: 1-Char/data)"
    )

    args = parser.parse_args()

    # Assign configurations from args
    prompt_input = args.prompt
    out_dir = args.out_dir
    data_dir = args.data_dir
    embedding_dim = 128
    n_heads = 4
    n_layers = 4
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load the weights first to see how large the positional embedding matrix actually is
    weights_path = os.path.join(out_dir, 'completion_model.pth')
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"Could not find model weights at {weights_path}")
        
    checkpoint = torch.load(weights_path, map_location=device)

    # Dynamically extract the maximum sequence length the model supports
    max_supported_len = checkpoint['position_embedding.weight'].shape[0]

    # --- 2. Load Metadata Vocabulary ---
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

    # --- 5. Autoregressive Completion Function ---
    def complete_sequence(prompt_str, max_new_tokens=20):
        if not prompt_str.endswith("="):
            if "=" in prompt_str:
                prompt_str = prompt_str.split("=")[0] + "="
            else:
                prompt_str += "="
                
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

    # --- 6. Execution & Clean Output ---
    prediction = complete_sequence(prompt_input)
    print(prediction)

if __name__ == "__main__":
    main()