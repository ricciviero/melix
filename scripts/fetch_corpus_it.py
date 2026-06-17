"""Scarica un SUBSET di Wikipedia in italiano (streaming) → un .txt per il tokenizer.

Streaming da Hugging Face (`load_dataset(..., streaming=True)`): NON scarica l'intero
dataset sul SSD da 512GB, legge a flusso e si ferma alla soglia. Output su disco esterno
(via melix/paths.py → MELIX_STORAGE_ROOT). Cache HF su HF_HOME (anch'essa su Archivio).

Uso:
    python -m scripts.fetch_corpus_it --max-mb 120
oppure:
    python scripts/fetch_corpus_it.py --max-mb 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Permette `python scripts/fetch_corpus_it.py` (aggiunge la repo root al path)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from datasets import load_dataset  # noqa: E402

from melix import paths  # noqa: E402


def fetch(max_mb: float, out_path: Path, dataset: str, config: str) -> None:
    paths.ensure()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    max_bytes = int(max_mb * 1024 * 1024)

    print(f"→ streaming {dataset} [{config}] fino a ~{max_mb:.0f} MB → {out_path}")
    ds = load_dataset(dataset, config, split="train", streaming=True)

    written = 0
    n_art = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for row in ds:
            text = (row.get("text") or "").strip()
            if not text:
                continue
            f.write(text)
            f.write("\n\n")
            written += len(text.encode("utf-8")) + 2
            n_art += 1
            if n_art % 2000 == 0:
                print(f"   {n_art:>6d} articoli · {written/1024/1024:6.1f} MB")
            if written >= max_bytes:
                break

    print(f"✓ scritti {n_art} articoli, {written/1024/1024:.1f} MB in {out_path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Corpus italiano (Wikipedia) in streaming per il tokenizer.")
    ap.add_argument("--max-mb", type=float, default=120.0, help="tetto dimensione corpus in MB")
    ap.add_argument("--dataset", default="wikimedia/wikipedia", help="dataset HF")
    ap.add_argument("--config", default="20231101.it", help="config/lingua del dataset")
    ap.add_argument("--out", default=None, help="output .txt (default: paths.DATASETS/wikipedia-it.txt)")
    args = ap.parse_args()

    out = Path(args.out) if args.out else (paths.DATASETS / "wikipedia-it.txt")
    fetch(args.max_mb, out, args.dataset, args.config)
