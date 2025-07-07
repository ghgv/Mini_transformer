import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return x

class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, T, C = x.shape
        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        qkv = qkv.reshape(B, T, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        Q, K, V = qkv[0], qkv[1], qkv[2]  # Each: (B, heads, T, head_dim)

        attn_scores = (Q @ K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        attn_weights = F.softmax(attn_scores, dim=-1)
        out = attn_weights @ V  # (B, heads, T, head_dim)
        out = out.transpose(1, 2).reshape(B, T, C)  # (B, T, C)
        return self.out_proj(out)

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
    def forward(self, x):
        return self.net(x)

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.attn = MultiHeadSelfAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class MiniTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=64, num_heads=4, d_ff=256, num_layers=2, max_len=128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        B, T = x.shape
        mask = torch.tril(torch.ones(T, T, device=x.device)).unsqueeze(0).unsqueeze(0)  # (1, 1, T, T)
        x = self.embed(x)
        x = self.pos_enc(x)
        for layer in self.layers:
            x = layer(x, mask)
        x = self.ln_f(x)
        logits = self.head(x)
        print("Embeddings shape:", x.shape)
        print("Embedding vector for first token:", x[0,0,:])
        print("Embedding vector for first token:", x[0,1,:])
        return logits

# Ejemplo de entrenamiento mínimo
if __name__ == "__main__":
    text = "el gato duerme en la casa"
    tokens = text.split()
    vocab = {w: i for i, w in enumerate(set(tokens))}
    vocab_size = len(vocab)
    indices = torch.tensor([vocab[w] for w in tokens])[None, :]  # (1, T)

    model = MiniTransformer(vocab_size, d_model=32, num_heads=4, d_ff=128, num_layers=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for step in range(500):
        logits = model(indices[:, :-1])
        loss = F.cross_entropy(logits.view(-1, vocab_size), indices[:, 1:].reshape(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 50 == 0:
            print(f"Step {step}, Loss: {loss.item():.4f}")

    # Autoregressive generation de 5 tokens
    generated = indices[:, :1]
    
    for _ in range(5):
        logits = model(generated)
        probs = F.softmax(logits[:, -1, :], dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        print("Next token",next_token)
        generated = torch.cat([generated, next_token], dim=1)
    idx_to_word = {i: w for w, i in vocab.items()}
    print("Generated:", ' '.join(idx_to_word[i.item()] for i in generated[0]))

    