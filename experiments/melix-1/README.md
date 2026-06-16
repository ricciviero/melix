# melix-1 — Proof of life

**Scopo**: chiudere l'intero giro della pipeline su scala minuscola, in **locale (Mac M1)**, gratis. Obiettivo didattico, non qualità.

- Modello: ~6M parametri (hidden 256 · 6 layer · 8/2 head · vocab 8k · block 512) — vedi `config.yaml`.
- Corpus: giocattolo (TinyStories, o un `.txt` italiano piccolo). In streaming, niente download enormi.
- Hardware: solo Mac (MPS/CPU). **Niente cloud.**

## Definition of Done
- [ ] tokenizer SentencePiece addestrato (`tokenizer/bpe.model`)
- [ ] shard di pretrain creati (uint16)
- [ ] **overfit-1-batch verde** (`python melix/model.py` → loss → ~0)
- [ ] pretrain breve con **loss che scende** (parte da ~ln(8000)≈9.0)
- [ ] `sample.py` genera testo plausibile (anche sgrammaticato: è ~6M)

## Come si lancia (quando i moduli sono implementati)
```bash
cd ~/Desktop/REPO/melix && source .venv/bin/activate
python melix/model.py                      # sanity: overfit-1-batch
# poi: tokenizer → data → train → sample, guidati dalla skill melix-llm-lab
```

## Esito (compilare a fine esperimento)
- loss finale: …
- sample d'esempio: …
- costo: 0 (locale)
- lezione imparata: …
