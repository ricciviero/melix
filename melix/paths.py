"""Percorsi di storage — dati pesanti su disco esterno, codice sull'interno.

Convenzione Melix:
- **Codice + venv** → SSD interno (questo repo).
- **Dati pesanti** (dataset, shard tokenizzati, checkpoint, cache HuggingFace, export)
  → disco esterno se disponibile (es. exFAT "Archivio"), via env `MELIX_STORAGE_ROOT`.

I moduli (data.py/train.py/export.py) usano queste costanti invece di path hardcoded.
Le variabili vengono da `.env` (vedi `.env.example`); senza disco esterno si ripiega su
`_local_storage/` dentro il repo (gitignored) — comodo per i test, NON per i dataset grandi.
"""

from __future__ import annotations

import os
from pathlib import Path

_FALLBACK = Path(__file__).resolve().parent.parent / "_local_storage"


def storage_root() -> Path:
    return Path(os.environ.get("MELIX_STORAGE_ROOT") or _FALLBACK)


def external_mounted() -> bool:
    """True se MELIX_STORAGE_ROOT è impostato E la cartella esiste (disco montato)."""
    root = os.environ.get("MELIX_STORAGE_ROOT")
    return bool(root) and Path(root).is_dir()


ROOT = storage_root()
HF_CACHE = Path(os.environ.get("HF_HOME") or ROOT / "hf-cache")
DATASETS = ROOT / "datasets"          # corpora grezzi/curati
DATA = Path(os.environ.get("MELIX_DATA_DIR") or ROOT / "data")          # shard uint16
CHECKPOINTS = Path(os.environ.get("MELIX_CKPT_DIR") or ROOT / "checkpoints")
EXPORTS = ROOT / "exports"            # ONNX/safetensors prima del push su HF
TOKENIZERS = ROOT / "tokenizers"      # bpe.model addestrati
LOGS = ROOT / "logs"                  # wandb/tensorboard offline


def ensure() -> None:
    """Crea le cartelle di storage se mancano (no-op se già presenti)."""
    for p in (HF_CACHE, DATASETS, DATA, CHECKPOINTS, EXPORTS, TOKENIZERS, LOGS):
        Path(p).mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    print(f"storage root : {ROOT}")
    print(f"esterno montato: {external_mounted()}")
    for name, p in [("hf-cache", HF_CACHE), ("datasets", DATASETS), ("data", DATA),
                    ("checkpoints", CHECKPOINTS), ("exports", EXPORTS),
                    ("tokenizers", TOKENIZERS), ("logs", LOGS)]:
        print(f"  {name:12s} → {p}  {'(esiste)' if Path(p).is_dir() else '(da creare)'}")
