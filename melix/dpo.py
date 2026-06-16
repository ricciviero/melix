"""Stadio 5 — DPO (Direct Preference Optimization), opzionale. [STUB]

Contratto: dpo(cfg) → su triple {prompt, chosen, rejected}, allinea alle preferenze.
beta ~0.1, LR ~1e-6 (poco sensibile alla scala), 1-4 epoche. Richiede un modello di
riferimento congelato (copia del modello SFT).

Tool consigliato: HF trl.DPOTrainer + ricette alignment-handbook. Dataset: UltraFeedback.
Spesso disattivato all'inizio (use_dpo=false): lecito fermarsi a SFT per le prime versioni di Melix.
Vedi references/pipeline.md §5.
"""

from __future__ import annotations

from melix.config import MelixConfig


def dpo(cfg: MelixConfig) -> None:
    raise NotImplementedError("DPO opzionale. Vedi references/pipeline.md §5.")
