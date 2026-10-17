from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml


@dataclass
class GroqConfig:
    gen_model: str
    translate_model: str
    rpm: int
    max_retries: int
    prompt_version: str


@dataclass
class Config:
    seed: int
    datasets: dict[str, int | None]
    retrievers: list[str]
    models: dict[str, str]
    groq: GroqConfig | dict[str, Any]
    noise_levels: list[float]
    top_k: int
    device: str
    batch_size: dict[str, int]


def load_config(path: str | Path = "configs/experiment.yaml") -> Config:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    groq_data = data.get("groq", {})
    groq_cfg = GroqConfig(
        gen_model=groq_data.get("gen_model", "TODO_check_console"),
        translate_model=groq_data.get("translate_model", "TODO_check_console"),
        rpm=groq_data.get("rpm", 20),
        max_retries=groq_data.get("max_retries", 3),
        prompt_version=groq_data.get("prompt_version", "v1"),
    )

    return Config(
        seed=data.get("seed", 42),
        datasets=data.get("datasets", {}),
        retrievers=data.get("retrievers", []),
        models=data.get("models", {}),
        groq=groq_cfg,
        noise_levels=data.get("noise_levels", []),
        top_k=data.get("top_k", 100),
        device=data.get("device", "auto"),
        batch_size=data.get("batch_size", {}),
    )
