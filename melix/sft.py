"""Stadio 4 — SFT (Supervised Fine-Tuning) su coppie prompt→risposta. [STUB]

Contratto: sft(cfg) → parte dal checkpoint di pretrain, fine-tune sul formato chat.
Loss mascherata sul SOLO testo dell'assistant. sft_epochs / sft_lr (più basso del pretrain;
nei modelli piccoli a volte più alto — testa). Template chat con token di ruolo.

Tool: loop tuo, oppure HF trl.SFTTrainer / mlx-lm (in locale per modelli piccoli).
Dati: SmolTalk, o un proprio dataset di coppie in italiano. Vedi references/pipeline.md §4.
"""

from __future__ import annotations

from melix.config import MelixConfig


def sft(cfg: MelixConfig) -> None:
    raise NotImplementedError("SFT sul checkpoint di pretrain. Vedi references/pipeline.md §4.")
