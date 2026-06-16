# Melix — un LLM costruito da zero

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE) · **Open source** (codice) · **Open weights** (pesi su Hugging Face)

**Melix** è un laboratorio di ricerca per progettare, addestrare e servire un Large Language Model **open**, **da zero**, versione dopo versione (`melix-1`, `melix-2`, …). Obiettivo doppio: **imparare** come nasce davvero un LLM e **produrre** modelli funzionanti, scalando di taglia a ogni iterazione. Codice su **GitHub**, pesi su **Hugging Face** — entrambi sotto **Apache-2.0**.

Architettura di riferimento: **decoder-only transformer stile Llama** (RMSNorm · RoPE · SwiGLU · GQA · tied embeddings), tokenizer SentencePiece BPE, export ONNX per il serving.

> Sviluppo assistito da **Claude Code** con la skill globale **`melix-llm-lab`** (in `~/.claude/skills/`), che fa da partner tecnico: conosce la pipeline, i tool locali/cloud e i vincoli dell'hardware.

## Hardware & filosofia

Macchina di sviluppo: **MacBook M1 Pro, 32 GB RAM, 512 GB SSD**. L'M1 **non** addestra modelli veri (niente CUDA, no bf16 nativo). Quindi:

- **Mac** = sviluppo del codice, proof-of-life, tokenizer, preprocessing in streaming, inferenza (MLX/ONNX), LoRA su modelli piccoli.
- **GPU cloud** (RunPod / Vast.ai / Lambda / Modal) = pretraining e fine-tuning su scala.

**Regola d'oro**: nessun run grande (o cloud a pagamento) prima di aver dimostrato la pipeline sul minuscolo (proof-of-life + *overfit di 1 batch*).

## La pipeline (8 stadi)

```
tokenizer → data → pretrain → SFT → DPO(opz.) → eval → export ONNX → serve
```

Dettagli e iperparametri di partenza: `docs/02-architettura-e-pipeline.md`.

## Struttura del repo

```
melix/
├── melix/                  # package: il codice condiviso (modello, training, dati…)
│   ├── config.py           # la "ricetta" come dataclass (architettura + iperparametri)
│   ├── model.py            # transformer stile Llama (v0 di riferimento)
│   ├── paths.py            # percorsi storage (dati su disco esterno via .env)
│   ├── tokenizer.py        # train/load SentencePiece          [stub → da implementare]
│   ├── data.py             # streaming + shard binari uint16   [stub]
│   ├── train.py            # loop di pretraining                [stub]
│   ├── sft.py              # fine-tuning supervisionato         [stub]
│   ├── dpo.py              # preferenze (opzionale)             [stub]
│   ├── sample.py           # generazione/inferenza             [stub]
│   └── export.py           # export ONNX                       [stub]
├── experiments/            # un esperimento per versione (solo config + note + esiti)
│   └── melix-1/            # v1: proof-of-life (~10M) sul Mac
├── eval/                   # harness di valutazione
├── serving/                # serving (FastAPI + onnxruntime)
├── scripts/                # setup mac/cloud, download dati
├── docs/                   # conoscenza, hardware/cloud, pipeline, roadmap
├── requirements-mac.txt    # ambiente locale (MLX, torch MPS)
└── requirements-cloud.txt  # ambiente GPU (CUDA, flash-attn)
```

Codice condiviso in `melix/`; ogni esperimento è **solo** un `config.yaml` + note in `experiments/melix-N/` (mai duplicare codice). Checkpoint e dataset **non** sono in git (vedi `.gitignore`).

## Quickstart (locale, M1)

```bash
cd ~/Desktop/REPO/melix
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements-mac.txt
# poi, con Claude Code + skill melix-llm-lab, si parte da experiments/melix-1
```

## Stato

🟢 Scaffold creato. Codice dei moduli `melix/` = **da implementare** (stub con specifica), guidato dalla skill. Inizia da `experiments/melix-1/README.md`.

## Dove studiare

`docs/00-conoscenza-e-risorse.md` raccoglie corsi, libri, repo, paper e dataset con i link.

## Riferimenti

- Guida operativa agenti (fonte di verità): [`CLAUDE.md`](CLAUDE.md)
- Skill partner: `melix-llm-lab` (globale, in `~/.claude/skills/`)
- Conoscenza e fonti: [`docs/00-conoscenza-e-risorse.md`](docs/00-conoscenza-e-risorse.md) · Hardware/cloud: [`docs/01-hardware-e-cloud.md`](docs/01-hardware-e-cloud.md) · Architettura/pipeline: [`docs/02-architettura-e-pipeline.md`](docs/02-architettura-e-pipeline.md) · Roadmap: [`docs/03-roadmap-versioni.md`](docs/03-roadmap-versioni.md)
- Decision log: `.claude/decisions/` · Memoria di progetto: `.claude/memory/`

## Licenza & open source

- **Licenza**: [Apache-2.0](LICENSE) per **codice e pesi** (vedi `NOTICE`).
- **Distribuzione**: codice su GitHub, **pesi su Hugging Face** (non in git — sono GB). Guida e checklist di release in [`docs/04-open-source-e-release.md`](docs/04-open-source-e-release.md); model card in [`docs/model-card-template.md`](docs/model-card-template.md); pubblicazione pesi con `scripts/push_to_hub.sh`.
- **Contribuire**: [`CONTRIBUTING.md`](CONTRIBUTING.md). **Citazione**: [`CITATION.cff`](CITATION.cff).
- ⚠️ **Dati**: i dataset di terzi restano sotto la loro licenza (citala, non riassegnarla). Dati raccolti da utenti → non pubblicabili senza basi legali (consenso/GDPR). Vedi `docs/04`.
