import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden, n_head):
        super().__init__()
        assert hidden % n_head == 0
        self.n_head = n_head
        self.d_k = hidden // n_head
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)

    def forward(self, x, mask=None):
        B, T, H = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_k).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            mask = mask[:, None, None, :]
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)

        out = attn @ v
        out = out.transpose(1, 2).contiguous().view(B, T, H)
        return self.out(out)

class EncoderLayer(nn.Module):
    def __init__(self, hidden, n_head, ff):
        super().__init__()
        self.attn = MultiHeadAttention(hidden, n_head)
        self.ln1 = nn.LayerNorm(hidden)
        self.ffn = nn.Sequential(
            nn.Linear(hidden, ff),
            nn.GELU(),
            nn.Linear(ff, hidden),
        )
        self.ln2 = nn.LayerNorm(hidden)

    def forward(self, x, mask=None):
        x = self.ln1(x + self.attn(x, mask))
        x = self.ln2(x + self.ffn(x))
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, hidden=768, n_layer=12, n_head=12, ff=3072):
        super().__init__()
        self.layers = nn.ModuleList([EncoderLayer(hidden, n_head, ff) for _ in range(n_layer)])

    def forward(self, x, mask=None):
        for layer in self.layers:
            x = layer(x, mask)
        return x

if __name__ == "__main__":
    x = torch.randn(2, 10, 768)
    mask = torch.ones(2, 10).bool()
    model = TransformerEncoder()
    out = model(x, mask)
    print(out.shape)
