"""Stadio 2 — Dati: streaming + shard binari uint16 per il pretrain. [STUB]

Contratto:
- prepare_pretrain(dataset, tokenizer, out_dir): stream del corpus (mai TB sul SSD!) →
  tokenizza → concatena gli id in shard binari `uint16` (np.memmap). Vocab < 65536 → uint16.
- get_batch(split, block_size, batch_size, device): campiona (x, y) dai shard via memmap,
  dove y = x shiftato di 1 (next-token).
- load_sft_jsonl / load_dpo_jsonl: legge i dataset di fine-tuning.

VINCOLO SSD 512GB: in locale usa datasets.load_dataset(..., streaming=True) su un subset;
sul cloud scarica gli shard sul volume della macchina, non sul Mac. Vedi references/hardware-cloud.md.
Per l'italiano: FineWeb-2 (config 'ita'). Riferimento formato shard: nanoGPT/nanochat.
"""

from __future__ import annotations


def prepare_pretrain(dataset_name: str, tokenizer, out_dir: str, streaming: bool = True) -> None:
    raise NotImplementedError(
        "Stream da HuggingFace datasets → tokenizza → np.memmap uint16 in shard. "
        "Vedi references/pipeline.md §2 e il vincolo SSD in hardware-cloud.md."
    )


def get_batch(split: str, block_size: int, batch_size: int, device: str = "cpu"):
    raise NotImplementedError("Campiona (x, y) dai shard memmap; y = x shiftato di 1.")
