from pathlib import Path

from rlhf_chatbot.config import PathSettings, load_settings


def test_default_configuration_loads() -> None:
    settings = load_settings()

    assert settings.models.policy_base == "gpt2"
    assert settings.ppo.clip_epsilon == 0.2
    assert settings.paths.ppo_checkpoint.endswith("ppo")


def test_relative_project_path_is_resolved(tmp_path: Path) -> None:
    paths = PathSettings(dataset="data/example.parquet")

    assert paths.resolve(paths.dataset, tmp_path) == tmp_path / "data/example.parquet"


def test_custom_configuration_overrides_defaults(tmp_path: Path) -> None:
    config = tmp_path / "experiment.toml"
    config.write_text(
        """
seed = 7

[models]
policy_base = "gpt2"
reward_base = "roberta-base"
max_length = 256
max_new_tokens = 32

[ppo]
learning_rate = 1e-6
update_epochs = 2
clip_epsilon = 0.1
kl_coefficient = 0.01
value_coefficient = 0.1
max_grad_norm = 0.5
""",
        encoding="utf-8",
    )

    settings = load_settings(config)

    assert settings.seed == 7
    assert settings.models.max_length == 256
    assert settings.ppo.clip_epsilon == 0.1
