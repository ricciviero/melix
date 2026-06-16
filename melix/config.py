"""Config = ricetta dell'esperimento (architettura + iperparametri di ogni stadio).

Un solo oggetto che cattura TUTTO ciò che serve a riprodurre un modello
(stesso principio dei config-ricetta di nanoGPT/litgpt). Si serializza in YAML (uno per cartella
`experiments/melix-N/config.yaml`). Nessuna dipendenza pesante: solo dataclass + pyyaml.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class MelixConfig:
    # --- identità ---
    name: str = "melix-1"
    seed: int = 1337  # riproducibilità: sempre fisso

    # --- architettura (decoder-only stile Llama) ---
    vocab_size: int = 8_000
    hidden_size: int = 256        # n_embd
    num_layers: int = 6           # n_layer
    num_heads: int = 8            # teste query
    num_kv_heads: int = 2         # teste KV (GQA: < num_heads). num_heads % num_kv_heads == 0
    intermediate_size: int = 0    # FFN; se 0 → calcolato da ff_mult (SwiGLU)
    ff_mult: float = 2.6666667    # moltiplicatore FFN (Llama ~8/3)
    block_size: int = 512         # context length
    rope_theta: float = 10_000.0
    dropout: float = 0.0
    tied_embeddings: bool = True
    norm_eps: float = 1e-5

    # --- pretraining ---
    pt_micro_batch: int = 16
    pt_grad_accum: int = 8        # batch efficace = micro_batch * grad_accum
    pt_max_steps: int = 5_000
    pt_lr: float = 4e-4
    pt_min_lr: float = 4e-5
    pt_warmup_steps: int = 200
    pt_weight_decay: float = 0.1
    pt_betas: tuple[float, float] = (0.9, 0.95)
    pt_grad_clip: float = 1.0
    pt_eval_interval: int = 250
    pt_ckpt_interval: int = 500

    # --- SFT ---
    sft_micro_batch: int = 4
    sft_grad_accum: int = 8
    sft_epochs: int = 3
    sft_lr: float = 1.5e-5
    sft_warmup_steps: int = 50

    # --- DPO (opzionale) ---
    use_dpo: bool = False
    dpo_beta: float = 0.1
    dpo_lr: float = 1e-6
    dpo_epochs: int = 1

    # --- export ---
    onnx_opset: int = 18

    # --- percorsi (relativi alla cartella dell'esperimento o a un volume) ---
    paths: dict = field(default_factory=lambda: {
        "tokenizer": "tokenizer/bpe.model",
        "pretrain_data": "data/pretrain",      # shard binari uint16
        "sft_data": "data/sft.jsonl",
        "dpo_data": "data/dpo.jsonl",
        "checkpoints": "checkpoints",
    })

    def __post_init__(self) -> None:
        if self.num_heads % self.num_kv_heads != 0:
            raise ValueError("num_heads deve essere multiplo di num_kv_heads (GQA).")
        if self.intermediate_size == 0:
            # SwiGLU: arrotonda a multiplo di 64 per efficienza
            raw = int(self.ff_mult * self.hidden_size)
            self.intermediate_size = ((raw + 63) // 64) * 64

    @property
    def head_dim(self) -> int:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError("hidden_size deve essere multiplo di num_heads.")
        return self.hidden_size // self.num_heads

    def estimate_params(self) -> int:
        """Stima dei parametri trainabili (GQA-aware, tied embeddings).
        Per la verifica si confronta col conteggio reale del modello (model.num_params())."""
        h, L, V = self.hidden_size, self.num_layers, self.vocab_size
        kv = self.num_kv_heads * self.head_dim
        # attn: Wq (h*h) + Wk,Wv (h*kv each) + Wo (h*h)
        attn = 2 * h * h + 2 * h * kv
        # SwiGLU: gate+up (h*inter each) + down (inter*h)
        mlp = 3 * h * self.intermediate_size
        per_layer = attn + mlp + 2 * h  # +2 RMSNorm (gain)
        emb = V * h  # tied → contata una volta
        return emb + L * per_layer + h  # + final norm

    # --- serializzazione YAML ---
    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        import yaml  # dipendenza locale, non al top
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f, sort_keys=False, allow_unicode=True)

    @classmethod
    def from_yaml(cls, path: str) -> "MelixConfig":
        import yaml
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if "pt_betas" in data and isinstance(data["pt_betas"], list):
            data["pt_betas"] = tuple(data["pt_betas"])
        return cls(**data)


if __name__ == "__main__":
    # smoke: stampa la stima parametri della config di default (melix-1)
    c = MelixConfig()
    print(f"{c.name}: ~{c.estimate_params()/1e6:.1f}M parametri "
          f"(hidden={c.hidden_size}, layer={c.num_layers}, "
          f"head={c.num_heads}/{c.num_kv_heads}, vocab={c.vocab_size})")
