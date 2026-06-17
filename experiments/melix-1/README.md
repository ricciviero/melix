# melix-1 — Proof of life

**Scopo**: chiudere l'intero giro della pipeline su scala minuscola, in **locale (Mac M1)**, gratis. Obiettivo didattico, non qualità.

- Modello: ~6M parametri (hidden 256 · 6 layer · 8/2 head · vocab 8k · block 512) — vedi `config.yaml`.
- Corpus: giocattolo (TinyStories, o un `.txt` italiano piccolo). In streaming, niente download enormi.
- Hardware: solo Mac (MPS/CPU). **Niente cloud.**

## Definition of Done
- [x] tokenizer SentencePiece addestrato (italiano, vocab 8000) — iter. 02, 2026-06-17
- [x] shard di pretrain creati (uint16) — iter. 03, Wikipedia IT 32,5M token
- [x] **overfit-1-batch verde** (`python -m melix.model` → loss → ~0) — fatto 2026-06-16, loss 5.54→0.0145
- [x] pretrain breve con **loss che scende** (parte da ~ln(8000)≈9.0) — iter. 04, 9.03→5.48 (val 5.27)
- [ ] `sample.py` genera testo plausibile (anche sgrammaticato: è ~6M) — iter. 05

## Come si lancia (quando i moduli sono implementati)
```bash
cd ~/Desktop/REPO/melix && source .venv/bin/activate
python -m melix.model                      # sanity: overfit-1-batch (i moduli con import di pacchetto vanno con -m)
# poi: tokenizer → data → train → sample, guidati dalla skill melix-llm-lab
```

## Esito (compilare a fine esperimento)
- loss finale: …
- sample d'esempio: …
- costo: 0 (locale)
- lezione imparata: …
