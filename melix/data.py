"""Stadio 2 — Dati: corpus → shard binari uint16 per il pretrain.

Contratto:
- prepare_pretrain(corpus_path, tokenizer, out_dir): legge il corpus (già scaricato in
  streaming all'iter. 02 → mai TB sul SSD), tokenizza per documenti (eos tra documenti),
  concatena gli id in shard binari `uint16` (vocab < 65536). Split train/val deterministico.
- get_batch(split, block_size, batch_size, device): campiona (x, y) dagli shard via np.memmap,
  dove y = x shiftato di 1 (next-token).
- load_sft_jsonl: helper per il fine-tuning → implementato all'iter. 07 (SFT persona).

Formato shard: concatenazione di token id in `np.memmap` uint16 (stile nanoGPT): lettura
random O(1), memmap = niente caricamento dell'intero file in RAM al training.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from melix import paths

DTYPE = np.uint16  # valido solo se vocab < 65536


def _iter_docs(corpus_path: str):
    """Itera i documenti di un .txt separati da riga vuota (formato di fetch_corpus_it.py)."""
    doc: list[str] = []
    with open(corpus_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() == "":
                if doc:
                    yield " ".join(doc)
                    doc = []
            else:
                doc.append(line.strip())
        if doc:
            yield " ".join(doc)


def prepare_pretrain(
    corpus_path: str,
    tokenizer,
    out_dir: str | None = None,
    val_frac: float = 0.01,
) -> dict[str, int]:
    """Tokenizza il corpus → `train.bin` + `val.bin` (uint16) in `out_dir` (default paths.DATA)."""
    if tokenizer.vocab_size >= 65536:
        raise ValueError(
            f"vocab_size={tokenizer.vocab_size} ≥ 65536: uint16 non basta, usa uint32."
        )
    if not Path(corpus_path).is_file():
        raise FileNotFoundError(f"Corpus non trovato: {corpus_path}")

    out = Path(out_dir) if out_dir else paths.DATA
    out.mkdir(parents=True, exist_ok=True)

    # Tokenizza per documenti (eos a fine documento), accumula in array piccoli per non
    # gonfiare la RAM con una lista di int Python.
    chunks: list[np.ndarray] = []
    n_docs = 0
    for doc in _iter_docs(corpus_path):
        ids = tokenizer.encode(doc, eos=True)
        chunks.append(np.asarray(ids, dtype=DTYPE))
        n_docs += 1
    if not chunks:
        raise ValueError(f"Corpus senza documenti utili: {corpus_path}")

    arr = np.concatenate(chunks)
    n_val = int(len(arr) * val_frac)
    if n_val == 0 or len(arr) - n_val == 0:
        raise ValueError(f"Corpus troppo piccolo per val_frac={val_frac} (token totali={len(arr)})")

    # Split deterministico: la coda va in validation, il resto in train.
    train, val = arr[:-n_val], arr[-n_val:]
    train.tofile(out / "train.bin")
    val.tofile(out / "val.bin")

    return {"docs": n_docs, "tokens": int(len(arr)), "train": int(len(train)), "val": int(len(val))}


def get_batch(
    split: str,
    block_size: int,
    batch_size: int,
    device: str = "cpu",
    data_dir: str | None = None,
    generator: torch.Generator | None = None,
):
    """Campiona (x, y) dallo shard `<split>.bin`; y = x shiftato di 1. memmap → niente full-load."""
    d = Path(data_dir) if data_dir else paths.DATA
    shard = d / f"{split}.bin"
    if not shard.is_file():
        raise FileNotFoundError(
            f"Shard non trovato: {shard}. Esegui prima prepare_pretrain() "
            f"(disco esterno montato? {paths.external_mounted()})."
        )

    data = np.memmap(shard, dtype=DTYPE, mode="r")
    if len(data) <= block_size + 1:
        raise ValueError(f"Shard {split} troppo corto ({len(data)}) per block_size={block_size}")

    ix = torch.randint(len(data) - block_size - 1, (batch_size,), generator=generator)
    # astype(int64): i target di cross_entropy devono essere long; uint16 va promosso
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    if device != "cpu":
        # niente pin_memory(): è una primitiva CUDA, su MPS/Apple Silicon non si applica
        x, y = x.to(device), y.to(device)
    return x, y


def load_sft_jsonl(path: str, tokenizer, block_size: int):
    """Helper SFT (formato chat con loss mascherata sull'assistant). → iterazione 07."""
    raise NotImplementedError(
        "Implementato all'iter. 07 (SFT persona): legge JSONL chat, target -1 sul prompt."
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Prepara gli shard di pretrain (uint16) e fa lo smoke.")
    ap.add_argument("--corpus", default=None, help="path del .txt (default: smoke su mini-corpus sintetico)")
    ap.add_argument("--block", type=int, default=64)
    ap.add_argument("--batch", type=int, default=4)
    args = ap.parse_args()

    from melix import tokenizer as tok_mod

    tok = tok_mod.load(str(paths.TOKENIZERS / "bpe.model"))

    if args.corpus:
        corpus = args.corpus
        out_dir = None
    else:
        # Smoke autosufficiente: mini-corpus sintetico in una dir temporanea dedicata.
        out_dir = str(paths.DATA / "_smoke")
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        corpus = str(Path(out_dir) / "mini.txt")
        frasi = ["Melissa ama i gattini e il caffè la mattina."] * 200
        Path(corpus).write_text("\n\n".join(frasi), encoding="utf-8")

    stats = prepare_pretrain(corpus, tok, out_dir=out_dir, val_frac=0.1 if not args.corpus else 0.01)
    print(f"✓ shard preparati: {stats}")

    x, y = get_batch("train", args.block, args.batch, data_dir=out_dir)
    print(f"✓ get_batch('train'): x={tuple(x.shape)} dtype={x.dtype}  y={tuple(y.shape)}")
    shift_ok = torch.equal(x[:, 1:], y[:, :-1])
    print(f"✓ y == x shiftato di 1: {shift_ok}")
    print(f"   x[0,:8] = {x[0,:8].tolist()}")
    print(f"   y[0,:8] = {y[0,:8].tolist()}")
