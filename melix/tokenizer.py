"""Stadio 1 — Tokenizer (SentencePiece BPE, byte_fallback). Corpus ITALIANO.

Contratto:
- train(corpus_path, model_prefix, vocab_size): allena SentencePiece su un CAMPIONE del
  corpus → scrive `<prefix>.model` + `<prefix>.vocab`. Token speciali: pad=0 unk=1 bos=2 eos=3.
- load(model_path) → oggetto con encode(str)->list[int] e decode(list[int])->str.

Per capire l'algoritmo da zero: karpathy/minbpe (vedi docs/00). In produzione si usa
`sentencepiece` (robusto, byte_fallback per coprire qualsiasi carattere fuori vocabolario).

Esecuzione (NB: import di package → si lancia con `-m`):
    python -m melix.tokenizer --corpus /Volumes/Archivio/melix/datasets/wikipedia-it.txt
"""

from __future__ import annotations

import argparse
import os

import sentencepiece as spm

from melix import paths

# Token speciali con id fissi (coerenti con data.py / sft.py futuri)
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


def train(corpus_path: str, model_prefix: str | None = None, vocab_size: int = 8000) -> str:
    """Allena un tokenizer SentencePiece BPE sul corpus e ritorna il path del `.model`."""
    if not os.path.isfile(corpus_path):
        raise FileNotFoundError(
            f"Corpus non trovato: {corpus_path}. "
            f"Generane uno con scripts/fetch_corpus_it.py (Wikipedia IT in streaming)."
        )
    if os.path.getsize(corpus_path) == 0:
        raise ValueError(f"Corpus vuoto: {corpus_path}")

    paths.ensure()
    prefix = model_prefix or str(paths.TOKENIZERS / "bpe")

    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        byte_fallback=True,              # copre qualsiasi carattere non in vocabolario
        character_coverage=0.9995,       # adatto a lingue con accenti (it)
        pad_id=PAD_ID, unk_id=UNK_ID, bos_id=BOS_ID, eos_id=EOS_ID,
        pad_piece="<pad>", unk_piece="<unk>", bos_piece="<bos>", eos_piece="<eos>",
        input_sentence_size=2_000_000,
        shuffle_input_sentence=True,
        num_threads=os.cpu_count() or 4,
    )
    return prefix + ".model"


class Tokenizer:
    """Wrapper sottile su SentencePieceProcessor: encode/decode + vocab_size."""

    def __init__(self, model_path: str) -> None:
        self.sp = spm.SentencePieceProcessor(model_file=model_path)
        self.model_path = model_path

    def encode(self, text: str, bos: bool = False, eos: bool = False) -> list[int]:
        ids = self.sp.encode(text, out_type=int)
        if bos:
            ids = [BOS_ID] + ids
        if eos:
            ids = ids + [EOS_ID]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()


def load(model_path: str) -> Tokenizer:
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Tokenizer non trovato: {model_path}. Esegui prima train().")
    return Tokenizer(model_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Allena e verifica il tokenizer BPE italiano di Melix.")
    ap.add_argument("--corpus", required=True, help="path del .txt di corpus (Wikipedia IT)")
    ap.add_argument("--vocab", type=int, default=8000, help="dimensione del vocabolario")
    ap.add_argument("--prefix", default=None, help="prefisso output (default: paths.TOKENIZERS/bpe)")
    args = ap.parse_args()

    print(f"→ addestro tokenizer BPE (vocab={args.vocab}) su {args.corpus}")
    model_file = train(args.corpus, model_prefix=args.prefix, vocab_size=args.vocab)
    print(f"✓ salvato: {model_file}")

    tok = load(model_file)
    print(f"✓ vocab_size reale: {tok.vocab_size}")

    # Smoke: roundtrip decode(encode(x)) == x su frasi italiane (con accenti)
    prove = [
        "Ciao, come stai oggi?",
        "L'università è perché così è più però città.",
        "Melissa ride sempre quando parliamo di gattini 😺",
        "Nel mezzo del cammin di nostra vita.",
    ]
    print("→ smoke roundtrip:")
    ok = True
    for s in prove:
        ids = tok.encode(s)
        back = tok.decode(ids)
        match = back == s
        ok = ok and match
        print(f"   [{'OK' if match else 'KO'}] {len(ids):3d} tok · {s!r}")
        if not match:
            print(f"        ↳ ricostruito: {back!r}")
    print("✓ roundtrip verde" if ok else "✗ roundtrip FALLITO (controlla byte_fallback/normalizzazione)")
