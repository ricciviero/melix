"""Stadio 3 — Pretraining (next-token prediction). [STUB]

Contratto: pretrain(cfg: MelixConfig) → addestra Melix sui shard, salva checkpoint.
Elementi: AdamW(betas, weight_decay) · LR cosine con warmup (pt_warmup→pt_max_steps,
pt_lr→pt_min_lr) · grad accumulation (pt_grad_accum) · grad clip (pt_grad_clip) ·
eval ogni pt_eval_interval · checkpoint ogni pt_ckpt_interval · seed fisso (cfg.seed).

PRIMA del run vero: forward su batch finto + OVERFIT DI 1 BATCH (loss→~0). Su Mac per il
proof-of-life (melix-1); su GPU cloud per le versioni successive (stima il costo prima).
Riferimento: nanoGPT/train.py. Vedi references/pipeline.md §3.
"""

from __future__ import annotations

from melix.config import MelixConfig


def pretrain(cfg: MelixConfig) -> None:
    raise NotImplementedError(
        "Loop di pretraining. Validare prima con overfit-1-batch (model.py __main__ lo fa). "
        "Iperparametri di partenza in references/pipeline.md §3."
    )
