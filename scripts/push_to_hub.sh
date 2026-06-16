#!/usr/bin/env bash
# Pubblica i PESI di una versione Melix su Hugging Face Hub (open weights).
# I pesi NON stanno su GitHub (sono GB): GitHub = codice, HF = pesi. Vedi docs/04.
#
# Prerequisiti: pip install huggingface_hub ; huggingface-cli login (token in env, NON committato)
# Uso: ./scripts/push_to_hub.sh <org/melix-N> <cartella_con_pesi>
set -euo pipefail

REPO_ID="${1:?Uso: push_to_hub.sh <org/melix-N> <cartella_pesi>}"
WEIGHTS_DIR="${2:?Manca la cartella con i pesi (es. experiments/melix-N/export)}"

if [ -z "${HF_TOKEN:-}" ] && [ ! -f "$HOME/.cache/huggingface/token" ]; then
  echo "⚠️  Nessun token HF trovato. Esegui 'huggingface-cli login' o esporta HF_TOKEN (mai in git)."
  exit 1
fi

echo "→ Verifica: la model card (README.md) è presente nella cartella?"
if [ ! -f "$WEIGHTS_DIR/README.md" ]; then
  echo "⚠️  Manca $WEIGHTS_DIR/README.md (model card). Copia docs/model-card-template.md e compilala prima di pubblicare."
  exit 1
fi

echo "→ Creo (se serve) il repo $REPO_ID e carico $WEIGHTS_DIR"
python - "$REPO_ID" "$WEIGHTS_DIR" <<'PY'
import sys
from huggingface_hub import HfApi
repo_id, folder = sys.argv[1], sys.argv[2]
api = HfApi()
api.create_repo(repo_id, repo_type="model", exist_ok=True)
api.upload_folder(repo_id=repo_id, folder_path=folder, repo_type="model")
print(f"✅ Pubblicato: https://huggingface.co/{repo_id}")
PY
echo "Ricorda: tagga la versione su GitHub (git tag melix-N) per legare codice ↔ pesi."
