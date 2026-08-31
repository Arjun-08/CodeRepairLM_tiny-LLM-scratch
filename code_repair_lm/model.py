import math
from typing import Optional

import torch
import torch.nn as nn


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x.pow(3))))


class LayerNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

    def forward(self, x):
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, unbiased=False, keepdim=True)
        x_norm = (x - mean) / torch.sqrt(var + self.eps)
        return self.gamma * x_norm + self.beta


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0):
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError("d_model must be divisible by n_head")
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        b, t, c = x.shape
        q = self.q_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(b, t, self.n_head, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attn_mask is None:
            attn_mask = torch.tril(torch.ones(t, t, device=x.device)).view(1, 1, t, t)
        else:
            attn_mask = attn_mask.to(x.device)
        scores = scores.masked_fill(attn_mask == 0, -1e9)
        probs = torch.softmax(scores, dim=-1)
        probs = self.dropout(probs)
        out = torch.matmul(probs, v)
        out = out.transpose(1, 2).contiguous().view(b, t, c)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(d_model, ff_dim)
        self.act = GELU()
        self.fc2 = nn.Linear(ff_dim, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, ff_dim: int, dropout: float = 0.0):
        super().__init__()
        self.self_attn = CausalSelfAttention(d_model, n_head, dropout)
        self.ln1 = LayerNorm(d_model)
        self.ffn = FeedForward(d_model, ff_dim, dropout)
        self.ln2 = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.ln1(x)
        x = self.self_attn(x)
        x = residual + self.dropout(x)
        residual = x
        x = self.ln2(x)
        x = self.ffn(x)
        x = residual + self.dropout(x)
        return x


class CodeRepairLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        block_size: int = 256,
        ff_dim: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_head = n_head
        self.n_layer = n_layer
        self.block_size = block_size
        self.ff_dim = ff_dim
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(block_size, d_model)
        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_head, ff_dim, dropout) for _ in range(n_layer)]
        )
        self.final_ln = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, max_seq_len: Optional[int] = None):
        if max_seq_len is None:
            max_seq_len = input_ids.shape[1]
        positions = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        x = self.dropout(x)
        seq_len = input_ids.shape[1]
        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device)).unsqueeze(0).unsqueeze(0)
        for layer in self.layers:
            x = layer(x)
        x = self.final_ln(x)
        logits = self.lm_head(x)
        return logits

    def generate(self, input_ids, max_new_tokens=32, temperature=1.0):
        self.eval()
        with torch.no_grad():
            generated = input_ids.clone()
            for _ in range(max_new_tokens):
                if generated.shape[1] > self.block_size:
                    generated = generated[:, -self.block_size:]
                logits = self(generated)
                next_logits = logits[:, -1, :] / temperature
                probs = torch.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                generated = torch.cat([generated, next_token], dim=1)
                if next_token.item() == 3:
                    break
            return generated

    def get_num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
