"""Typed configuration and repository path handling."""

from __future__ import annotations

import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class PathSettings:
    dataset: str = "data/raw/ultra_feedback.parquet"
    evaluation_prompts: str = "data/evaluation/prompts.xlsx"
    sft_checkpoint: str = "artifacts/models/sft"
    reward_checkpoint: str = "artifacts/models/reward"
    ppo_checkpoint: str = "artifacts/models/ppo"
    evaluation_figure: str = "docs/assets/evaluation.png"

    def resolve(self, value: str, root: Path = REPOSITORY_ROOT) -> Path:
        path = Path(value).expanduser()
        return path if path.is_absolute() else root / path


@dataclass(frozen=True)
class ModelSettings:
    policy_base: str = "gpt2"
    reward_base: str = "roberta-base"
    max_length: int = 512
    max_new_tokens: int = 128


@dataclass(frozen=True)
class SFTSettings:
    batch_size: int = 4
    learning_rate: float = 5e-5
    epochs: int = 1


@dataclass(frozen=True)
class RewardSettings:
    batch_size: int = 4
    learning_rate: float = 2e-5
    epochs: int = 1
    margin: float = 1.0


@dataclass(frozen=True)
class PPOSettings:
    learning_rate: float = 5e-6
    update_epochs: int = 4
    clip_epsilon: float = 0.2
    kl_coefficient: float = 0.005
    value_coefficient: float = 0.05
    max_grad_norm: float = 1.0


@dataclass(frozen=True)
class EvaluationSettings:
    judge_model: str = "gpt-4o-mini"
    prompt_column: str = "Prompts"


@dataclass(frozen=True)
class Settings:
    paths: PathSettings = field(default_factory=PathSettings)
    models: ModelSettings = field(default_factory=ModelSettings)
    sft: SFTSettings = field(default_factory=SFTSettings)
    reward: RewardSettings = field(default_factory=RewardSettings)
    ppo: PPOSettings = field(default_factory=PPOSettings)
    evaluation: EvaluationSettings = field(default_factory=EvaluationSettings)
    seed: int = 42


def _section(data: dict[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name, {})
    if not isinstance(value, dict):
        raise ValueError(f"Configuration section [{name}] must be a table.")
    return value


def load_settings(path: str | Path | None = None) -> Settings:
    """Load settings from TOML, falling back to version-controlled defaults."""

    config_path = Path(path) if path else REPOSITORY_ROOT / "configs/default.toml"
    if not config_path.exists():
        return Settings()

    with config_path.open("rb") as handle:
        data = tomllib.load(handle)

    return Settings(
        paths=PathSettings(**_section(data, "paths")),
        models=ModelSettings(**_section(data, "models")),
        sft=SFTSettings(**_section(data, "sft")),
        reward=RewardSettings(**_section(data, "reward")),
        ppo=PPOSettings(**_section(data, "ppo")),
        evaluation=EvaluationSettings(**_section(data, "evaluation")),
        seed=int(data.get("seed", 42)),
    )
