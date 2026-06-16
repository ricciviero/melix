"""Inferenza / generazione (autoregressiva). [STUB]

Contratto: generate(model, tokenizer, prompt, max_new_tokens, temperature, top_k, top_p, seed=None)
→ campiona token per token. temperature=0 → greedy (deterministico). Senza seed → output
NON deterministico per design (è normale: vedi docs/00). Per efficienza vera aggiungi KV-cache.

Logica di sampling standard (softmax + top-k + top-p). Vedi references/pipeline.md §7-8.
"""

from __future__ import annotations


def generate(model, tokenizer, prompt: str, max_new_tokens: int = 128,
             temperature: float = 0.8, top_k: int = 50, top_p: float = 0.95,
             seed: int | None = None) -> str:
    raise NotImplementedError(
        "Generazione autoregressiva con top-k/top-p. temperature=0 → greedy. "
        "Per riproducibilità: seed fisso. Vedi docs/00 (perché le risposte cambiano)."
    )
