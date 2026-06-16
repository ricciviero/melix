# Eval

Valutazione dei modelli Melix. Due livelli (vedi `docs/02` e skill `references/pipeline.md §6`):

- **Quantitativo**: loss di validazione + benchmark con `lm-evaluation-harness` (EleutherAI). Per modelli piccoli i punteggi sono bassi: conta la **tendenza tra versioni**.
- **Qualitativo**: un set FISSO di ~20 prompt (in `prompts.txt`, da creare) rilanciato a ogni versione per confronto a occhio — sempre gli stessi, così misuri il progresso.

Logga tutto (W&B o TensorBoard). Stub `eval.py` da implementare con la skill.
