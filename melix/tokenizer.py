"""Stadio 1 — Tokenizer (SentencePiece BPE, byte_fallback). [STUB — da implementare]

Contratto:
- train(corpus_path, model_prefix, vocab_size): allena SentencePiece su un CAMPIONE del
  corpus → scrive `<prefix>.model` + `<prefix>.vocab`. Token speciali: pad=0 unk=1 bos=2 eos=3.
- load(model_path) → oggetto con encode(str)->list[int] e decode(list[int])->str.

Implementazione consigliata: pacchetto `sentencepiece` (spm.SentencePieceTrainer.train).
Per capire l'algoritmo da zero: karpathy/minbpe (vedi docs/00). Non reinventare in produzione.
"""

from __future__ import annotations


def train(corpus_path: str, model_prefix: str, vocab_size: int = 8000) -> str:
    raise NotImplementedError(
        "Implementa con sentencepiece: SentencePieceTrainer.train("
        "input=corpus_path, model_prefix=model_prefix, vocab_size=vocab_size, "
        "model_type='bpe', byte_fallback=True, "
        "pad_id=0, unk_id=1, bos_id=2, eos_id=3). Vedi references/pipeline.md §1."
    )


def load(model_path: str):
    raise NotImplementedError("Implementa: ritorna un wrapper con encode/decode su spm.SentencePieceProcessor.")
