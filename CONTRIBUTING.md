# Contribuire a Melix

Grazie per l'interesse! Melix è un laboratorio open source (Apache-2.0) per costruire un LLM da zero.

## Come iniziare
1. Leggi `README.md` e `docs/` (architettura, pipeline, hardware/cloud).
2. Setup locale: `./scripts/setup_mac.sh` (Apple Silicon) o `./scripts/setup_cloud.sh` (GPU).
3. La guida operativa per gli agenti AI e le convenzioni del progetto sono in [`CLAUDE.md`](CLAUDE.md).

## Convenzioni
- **Config = ricetta**: ogni esperimento è `experiments/melix-N/config.yaml` (cattura tutto + `seed`). Niente magic number nel codice.
- **Codice condiviso** in `melix/`; gli esperimenti contengono solo config + note + esiti.
- **Regola d'oro**: niente run grande/cloud senza proof-of-life + **overfit-1-batch** (loss → ~0).
- **Riproducibilità**: seed fisso, risultati documentati nel README dell'esperimento.

## Pull request
- Una PR = un cambiamento coerente. Spiega *cosa* e *perché*.
- Non committare **pesi, dataset, checkpoint, segreti** (`.gitignore` li esclude; i pesi vanno su Hugging Face — vedi `docs/04-open-source-e-release.md`).
- Per nuovo codice di modello/training: includi l'esito dell'overfit-1-batch.

## Dati
Rispetta le licenze dei dataset di terzi e **non** aggiungere dati personali/utente senza basi legali chiare (consenso/GDPR). Vedi `docs/04-open-source-e-release.md`.

## Licenza dei contributi
Contribuendo accetti che il tuo contributo sia rilasciato sotto **Apache-2.0**.
