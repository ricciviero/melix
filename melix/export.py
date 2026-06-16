"""Stadio 7 — Export ONNX (per il serving portabile). [STUB]

Contratto: export_onnx(model, cfg, out_path) → torch.onnx.export con opset 18, path attention
'math' (portabile, no kernel custom), input dinamici su batch e sequenza. Salva accanto il
config (provenance: il modello porta con sé la ricetta che l'ha generato).

Serving: FastAPI + onnxruntime + sampling (vedi `serving/`). In locale, alternativa: MLX.
Vedi references/pipeline.md §7.
"""

from __future__ import annotations

from melix.config import MelixConfig


def export_onnx(model, cfg: MelixConfig, out_path: str) -> None:
    raise NotImplementedError(
        "torch.onnx.export(opset=cfg.onnx_opset, dynamic_axes su batch/seq, attn 'math'). "
        "Salva il config accanto al .onnx. Vedi references/pipeline.md §7."
    )
