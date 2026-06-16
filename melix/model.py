"""Melix — modello transformer decoder-only stile Llama (RMSNorm · RoPE · SwiGLU · GQA).

⚠️ v0 DI RIFERIMENTO — DA VALIDARE al primo run con i test obbligatori (vedi sotto e
`docs/02-architettura-e-pipeline.md`):
  1) forward su batch finto → loss finita
  2) OVERFIT DI 1 BATCH → la loss deve crollare a ~0 (se no: bug masking/RoPE/GQA)

Implementazione didattica e compatta, orientata al training (niente KV-cache: quella
si aggiunge in `sample.py` per l'inferenza efficiente). Dipende solo da PyTorch.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from melix.config import MelixConfig


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return norm * self.weight


def build_rope_cache(head_dim: int, max_seq: int, theta: float, device=None):
    """Precalcola cos/sin per RoPE. Shape ritornata: (max_seq, head_dim)."""
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    t = torch.arange(max_seq, device=device).float()
    freqs = torch.outer(t, inv_freq)              # (max_seq, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)       # (max_seq, head_dim)
    return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_head, T, head_dim); cos/sin: (T, head_dim) → broadcast su B e teste
    cos = cos[None, None, :, :]
    sin = sin[None, None, :, :]
    return x * cos + rotate_half(x) * sin


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """GQA: ripete le teste KV per matchare le teste Q. x: (B, n_kv, T, hd)."""
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv, n_rep, T, hd)
        .reshape(B, n_kv * n_rep, T, hd)
    )


class Attention(nn.Module):
    def __init__(self, cfg: MelixConfig) -> None:
        super().__init__()
        self.n_head = cfg.num_heads
        self.n_kv = cfg.num_kv_heads
        self.hd = cfg.head_dim
        self.n_rep = self.n_head // self.n_kv
        self.wq = nn.Linear(cfg.hidden_size, self.n_head * self.hd, bias=False)
        self.wk = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.wv = nn.Linear(cfg.hidden_size, self.n_kv * self.hd, bias=False)
        self.wo = nn.Linear(self.n_head * self.hd, cfg.hidden_size, bias=False)
        self.dropout = cfg.dropout

    def forward(self, x, cos, sin):
        B, T, _ = x.shape
        q = self.wq(x).view(B, T, self.n_head, self.hd).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_kv, self.hd).transpose(1, 2)
        q = apply_rope(q, cos[:T], sin[:T])
        k = apply_rope(k, cos[:T], sin[:T])
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)
        # attenzione causale (path 'math'/SDPA, portabile per l'export ONNX)
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_head * self.hd)
        return self.wo(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: MelixConfig) -> None:
        super().__init__()
        self.w_gate = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w_up = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.w_down = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: MelixConfig) -> None:
        super().__init__()
        self.norm1 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.attn = Attention(cfg)
        self.norm2 = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class Melix(nn.Module):
    def __init__(self, cfg: MelixConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.drop = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.num_layers)])
        self.norm = RMSNorm(cfg.hidden_size, cfg.norm_eps)
        self.lm_head = nn.Linear(cfg.hidden_size, cfg.vocab_size, bias=False)
        if cfg.tied_embeddings:
            self.lm_head.weight = self.tok_emb.weight  # tied
        cos, sin = build_rope_cache(cfg.head_dim, cfg.block_size, cfg.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def num_params(self, trainable_only: bool = True) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad or not trainable_only)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        B, T = idx.shape
        assert T <= self.cfg.block_size, f"sequenza {T} > block_size {self.cfg.block_size}"
        x = self.drop(self.tok_emb(idx))
        cos, sin = self.rope_cos.to(x.dtype), self.rope_sin.to(x.dtype)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1
            )
        return logits, loss


if __name__ == "__main__":
    # SANITY CHECK v0: forward + overfit di 1 batch (la loss deve crollare).
    torch.manual_seed(0)
    cfg = MelixConfig(vocab_size=256, hidden_size=128, num_layers=2,
                      num_heads=4, num_kv_heads=2, block_size=64)
    model = Melix(cfg)
    print(f"Parametri: reale={model.num_params()/1e6:.3f}M · stima_config={cfg.estimate_params()/1e6:.3f}M")
    x = torch.randint(0, cfg.vocab_size, (2, 32))
    y = torch.randint(0, cfg.vocab_size, (2, 32))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(200):
        _, loss = model(x, y)
        opt.zero_grad(); loss.backward(); opt.step()
        if step % 50 == 0:
            print(f"step {step}: loss {loss.item():.4f}")
    print(f"loss finale {loss.item():.4f} (deve essere ~0 → modello OK)")
