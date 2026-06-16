# Serving

Servire un modello Melix esportato in ONNX con un'API HTTP: **FastAPI + onnxruntime** + sampling (softmax + top-k + top-p), streaming dei token. È il pattern standard per servire un decoder-only in produzione.

Alternativa rapida in locale (Mac): **MLX** (`mlx-lm`) per l'inferenza.

Promemoria: senza seed l'output è **non-deterministico per design** (vedi `docs/00`). Per risposte riproducibili in demo: `temperature=0` (greedy) o seed fisso.

Da implementare con la skill `melix-llm-lab` quando esiste il primo modello esportato (≥ melix-2/3).
