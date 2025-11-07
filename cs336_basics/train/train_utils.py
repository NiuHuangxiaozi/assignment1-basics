import yaml
from dataclasses import dataclass
from typing import Literal, Optional

@dataclass
class ModelConfig:
    vocab_size: int
    context_length: int
    d_model: int
    d_ff: int
    rope_theta: int
    num_layers: int
    num_heads: int

    def validate(self):
        assert self.vocab_size > 0, "vocab_size must be positive"
        assert self.context_length > 0, "context_length must be positive"
        assert self.d_model > 0 and self.d_ff > 0, "model dims must be positive"
        assert self.d_ff % 64 == 0, "d_ff should be multiple of 64 for GPU perf"
        assert self.num_layers > 0 and self.num_heads > 0, "layers/heads must be positive"

@dataclass
class OptimizerConfig:
    type: Literal["AdamW", "SGD", "Adam"]
    beta1: float
    beta2: float
    eps: float
    weight_decay: float

@dataclass
class TrainingConfig:
    exp_name: str
    epochs: int
    train_ids_path: str
    device: str
    save_path: str
    epoch_print_freq: int 
    iter_print_freq: int 
    optimizer: OptimizerConfig
    learning_rate: float
    warmup_steps: int
    batch_size: int
    warmup_steps: int
    cosine_cycle_steps: int
    learning_rate_min: float
    max_norm: float
    resume_from_checkpoint: bool
    checkpoint_path: str

    def validate(self):
        assert 0 < self.learning_rate < 1, "learning_rate seems invalid"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.batch_size > 0, "batch_size must be positive"
        assert self.warmup_steps >= 0, "warmup_steps must be non-negative"
        assert self.cosine_cycle_steps >= 0, "cosine_cycle_steps must be non-negative"
        assert self.learning_rate_min >= 0, "learning_rate_min must be non-negative"
        assert self.max_norm >= 0, "max_norm must be non-negative"

def load_model_config(path: str) -> ModelConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    cfg = ModelConfig(**data)
    cfg.validate()
    return cfg

def load_training_config(path: str) -> TrainingConfig:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    opt = OptimizerConfig(**data["optimizer"])
    data["optimizer"] = opt
    cfg = TrainingConfig(**data)
    cfg.validate()
    return cfg

# Usage 示例
model_cfg = load_model_config("configs/model_configs.yaml")
training_cfg = load_training_config("configs/exp_configs.yaml")

print(model_cfg)
print(training_cfg)
