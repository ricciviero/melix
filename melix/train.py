"""Stadio 3 — Pretraining (next-token prediction).

Loop minimale stile nanoGPT, su MPS (Apple Silicon) per il proof-of-life melix-1.
Iperparametri dal `config.yaml` (mai hardcoded). Checkpoint/log su disco esterno.

Elementi: AdamW(betas, weight_decay) · LR cosine con warmup · grad accumulation
(pt_grad_accum) · grad clip (pt_grad_clip) · eval ogni pt_eval_interval · checkpoint
ogni pt_ckpt_interval · seed fisso (cfg.seed).

REGOLA D'ORO: prima del run vero, OVERFIT DI 1 BATCH (loss→~0).
    python -m melix.train --overfit
    python -m melix.train --steps 300        # run breve, loss deve scendere da ~ln(vocab)
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from melix import data, paths
from melix.config import MelixConfig
from melix.model import Melix


def get_device(explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def cosine_lr(step: int, cfg: MelixConfig) -> float:
    """Warmup lineare → cosine decay da pt_lr a pt_min_lr fino a pt_max_steps."""
    if step < cfg.pt_warmup_steps:
        return cfg.pt_lr * (step + 1) / cfg.pt_warmup_steps
    if step >= cfg.pt_max_steps:
        return cfg.pt_min_lr
    ratio = (step - cfg.pt_warmup_steps) / max(1, cfg.pt_max_steps - cfg.pt_warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * ratio))  # 1 → 0
    return cfg.pt_min_lr + coeff * (cfg.pt_lr - cfg.pt_min_lr)


@torch.no_grad()
def evaluate(model: Melix, cfg: MelixConfig, device: str, eval_iters: int = 50) -> float:
    """Media della loss su `eval_iters` batch dello split val."""
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = data.get_batch("val", cfg.block_size, cfg.pt_micro_batch, device)
        _, loss = model(x, y)
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def save_ckpt(model: Melix, opt, step: int, cfg: MelixConfig, val_loss: float | None) -> Path:
    ckpt_dir = paths.CHECKPOINTS / cfg.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / f"ckpt_{step:06d}.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": opt.state_dict(),
            "step": step,
            "val_loss": val_loss,
            "config": cfg.to_dict(),
        },
        path,
    )
    # comodo per il resume / sample: un alias all'ultimo
    torch.save({"model": model.state_dict(), "step": step, "config": cfg.to_dict()},
               ckpt_dir / "latest.pt")
    return path


def overfit_one_batch(cfg: MelixConfig, device: str | None = None, steps: int = 300) -> float:
    """REGOLA D'ORO: un solo batch ripetuto → la loss deve crollare verso ~0."""
    device = get_device(device)
    torch.manual_seed(cfg.seed)
    model = Melix(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.pt_lr,
                            betas=cfg.pt_betas, weight_decay=0.0)
    x, y = data.get_batch("train", cfg.block_size, cfg.pt_micro_batch, device)
    print(f"[overfit-1-batch] device={device} · ln(vocab)={math.log(cfg.vocab_size):.3f}")
    loss_val = float("nan")
    for step in range(steps):
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        loss_val = loss.item()
        if step % (steps // 6) == 0 or step == steps - 1:
            print(f"  step {step:4d}: loss {loss_val:.4f}")
    print(f"[overfit-1-batch] loss finale {loss_val:.4f} "
          f"({'OK → loop corretto' if loss_val < 0.5 else 'KO → bug nel loop/loss'})")
    return loss_val


def pretrain(
    cfg: MelixConfig,
    device: str | None = None,
    max_steps: int | None = None,
    log_interval: int = 20,
) -> Path:
    """Loop di pretraining vero. Ritorna il path dell'ultimo checkpoint."""
    device = get_device(device)
    steps_total = max_steps if max_steps is not None else cfg.pt_max_steps
    torch.manual_seed(cfg.seed)

    model = Melix(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.pt_lr,
                            betas=cfg.pt_betas, weight_decay=cfg.pt_weight_decay)

    n_params = model.num_params() / 1e6
    print(f"[pretrain] {cfg.name} · {n_params:.2f}M param · device={device} · "
          f"batch eff={cfg.pt_micro_batch * cfg.pt_grad_accum} · steps={steps_total}")
    print(f"[pretrain] loss attesa di partenza ≈ ln(vocab) = {math.log(cfg.vocab_size):.3f}")

    last_ckpt = None
    t0 = time.time()
    for step in range(steps_total):
        lr = cosine_lr(step, cfg)
        for g in opt.param_groups:
            g["lr"] = lr

        # grad accumulation su pt_grad_accum micro-batch
        opt.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.pt_grad_accum):
            x, y = data.get_batch("train", cfg.block_size, cfg.pt_micro_batch, device)
            _, loss = model(x, y)
            loss = loss / cfg.pt_grad_accum
            loss.backward()
            loss_accum += loss.item()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.pt_grad_clip)
        opt.step()

        if step % log_interval == 0 or step == steps_total - 1:
            dt = time.time() - t0
            print(f"  step {step:5d}/{steps_total} · loss {loss_accum:.4f} · lr {lr:.2e} · "
                  f"|g| {grad_norm:.2f} · {dt:.1f}s")

        if cfg.pt_eval_interval and step > 0 and step % cfg.pt_eval_interval == 0:
            vloss = evaluate(model, cfg, device)
            print(f"  ↳ [eval] step {step}: val_loss {vloss:.4f}")
        if cfg.pt_ckpt_interval and step > 0 and step % cfg.pt_ckpt_interval == 0:
            last_ckpt = save_ckpt(model, opt, step, cfg, None)
            print(f"  ↳ [ckpt] {last_ckpt}")

    vloss = evaluate(model, cfg, device)
    last_ckpt = save_ckpt(model, opt, steps_total, cfg, vloss)
    print(f"[pretrain] fine · val_loss {vloss:.4f} · checkpoint {last_ckpt}")
    return last_ckpt


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Pretraining di Melix (next-token).")
    ap.add_argument("--config", default="experiments/melix-1/config.yaml")
    ap.add_argument("--overfit", action="store_true", help="regola d'oro: overfit di 1 batch")
    ap.add_argument("--steps", type=int, default=None, help="override pt_max_steps (run breve)")
    ap.add_argument("--device", default=None, help="mps|cpu (default: auto)")
    args = ap.parse_args()

    cfg = (MelixConfig.from_yaml(args.config)
           if Path(args.config).is_file() else MelixConfig())

    if args.overfit:
        overfit_one_batch(cfg, device=args.device)
    else:
        pretrain(cfg, device=args.device, max_steps=args.steps)
