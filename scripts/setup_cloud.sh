#!/usr/bin/env bash
# Setup ambiente CLOUD su GPU NVIDIA (RunPod/Vast/Lambda/Modal). Idempotente.
# Esegui DENTRO la macchina cloud (immagine con CUDA), NON sul Mac.
set -euo pipefail
cd "$(dirname "$0")/.."

python -m pip install --upgrade pip
echo "→ installo requirements-cloud.txt"
pip install -r requirements-cloud.txt

echo "→ verifico CUDA"
python - <<'PY'
import torch
print("torch", torch.__version__, "| CUDA:", torch.cuda.is_available(),
      "| GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "—")
PY

# PROMEMORIA COSTO: stima ore × $/h prima di lanciare un run lungo.
# Checkpoint frequenti (le istanze spot si interrompono). SPEGNI l'istanza a fine job.
echo "OK. Ricorda: stima il costo, checkpoint frequenti, SPEGNI l'istanza a fine job."
