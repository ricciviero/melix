#!/usr/bin/env bash
# Setup ambiente LOCALE su Mac (Apple Silicon, M1 Pro). Idempotente.
set -euo pipefail
cd "$(dirname "$0")/.."

# Carica .env (percorsi storage su disco esterno + eventuali token). Mai committato.
if [ -f .env ]; then
  set -a; source .env; set +a
  echo "→ .env caricato"
  if [ -n "${MELIX_STORAGE_ROOT:-}" ] && [ ! -d "${MELIX_STORAGE_ROOT}" ]; then
    echo "⚠️  MELIX_STORAGE_ROOT=${MELIX_STORAGE_ROOT} non esiste: il disco esterno è montato?"
    echo "    (per i test puoi proseguire: si ripiega su _local_storage/ interno)"
  fi
else
  echo "ℹ️  Nessun .env: crea .env con MELIX_STORAGE_ROOT=/Volumes/Archivio/melix per usare il disco esterno."
fi

if [ ! -d .venv ]; then
  echo "→ creo .venv (python3)"
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip
echo "→ installo requirements-mac.txt"
pip install -r requirements-mac.txt

echo "→ verifico PyTorch MPS (Apple Silicon)"
python - <<'PY'
import torch
print("torch", torch.__version__, "| MPS disponibile:", torch.backends.mps.is_available())
PY

echo "→ smoke: stima parametri della config di default"
python melix/config.py

echo "→ percorsi di storage (dati pesanti su disco esterno):"
python melix/paths.py

echo "OK. Prossimo: 'python melix/model.py' per l'overfit-1-batch, poi experiments/melix-1."
