"""Command-line entry points for every reproducible pipeline stage."""

from __future__ import annotations

import argparse
import logging
from collections.abc import Sequence
from pathlib import Path

from rlhf_chatbot.config import REPOSITORY_ROOT, Settings, load_settings


def _parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default="configs/default.toml")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or mps")
    parser.add_argument("--limit", type=int, help="Optional sample limit for smoke runs")
    return parser


def _settings(config: str) -> Settings:
    path = Path(config)
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    return load_settings(path)


def _path(settings: Settings, value: str) -> Path:
    return settings.paths.resolve(value)


def _logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


def train_sft_main(argv: Sequence[str] | None = None) -> None:
    parser = _parser("Train the supervised GPT-2 policy.")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    settings = _settings(args.config)
    from rlhf_chatbot.training.sft import train_sft

    _logging()
    train_sft(
        settings,
        dataset_path=Path(args.dataset)
        if args.dataset
        else _path(settings, settings.paths.dataset),
        output_dir=Path(args.output)
        if args.output
        else _path(settings, settings.paths.sft_checkpoint),
        limit=args.limit,
        device=args.device,
    )


def train_reward_main(argv: Sequence[str] | None = None) -> None:
    parser = _parser("Train the pairwise RoBERTa reward model.")
    parser.add_argument("--dataset")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    settings = _settings(args.config)
    from rlhf_chatbot.training.reward import train_reward_model

    _logging()
    train_reward_model(
        settings,
        dataset_path=Path(args.dataset)
        if args.dataset
        else _path(settings, settings.paths.dataset),
        output_dir=Path(args.output)
        if args.output
        else _path(settings, settings.paths.reward_checkpoint),
        limit=args.limit,
        device=args.device,
    )


def train_ppo_main(argv: Sequence[str] | None = None) -> None:
    parser = _parser("Align the SFT policy with PPO and a learned reward.")
    parser.add_argument("--dataset")
    parser.add_argument("--sft-checkpoint")
    parser.add_argument("--reward-checkpoint")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    settings = _settings(args.config)
    from rlhf_chatbot.training.ppo import train_ppo

    _logging()
    train_ppo(
        settings,
        dataset_path=Path(args.dataset)
        if args.dataset
        else _path(settings, settings.paths.dataset),
        sft_checkpoint=(
            Path(args.sft_checkpoint)
            if args.sft_checkpoint
            else _path(settings, settings.paths.sft_checkpoint)
        ),
        reward_checkpoint=(
            Path(args.reward_checkpoint)
            if args.reward_checkpoint
            else _path(settings, settings.paths.reward_checkpoint)
        ),
        output_dir=Path(args.output)
        if args.output
        else _path(settings, settings.paths.ppo_checkpoint),
        limit=args.limit,
        device=args.device,
    )


def evaluate_main(argv: Sequence[str] | None = None) -> None:
    parser = _parser("Evaluate the SFT and PPO policies with an LLM judge.")
    parser.add_argument("--prompts")
    parser.add_argument("--output")
    args = parser.parse_args(argv)
    settings = _settings(args.config)

    import pandas as pd
    from dotenv import load_dotenv

    from rlhf_chatbot.evaluation import OpenAIJudge, evaluate_policies, plot_evaluation
    from rlhf_chatbot.models.policy import PolicyValueModel
    from rlhf_chatbot.models.reward import RewardModel

    _logging()
    load_dotenv(REPOSITORY_ROOT / ".env")
    prompt_path = (
        Path(args.prompts) if args.prompts else _path(settings, settings.paths.evaluation_prompts)
    )
    prompts = (
        pd.read_excel(prompt_path)[settings.evaluation.prompt_column].dropna().astype(str).tolist()
    )
    if args.limit is not None:
        prompts = prompts[: args.limit]

    sft = PolicyValueModel.from_checkpoint(
        _path(settings, settings.paths.sft_checkpoint), device=args.device
    )
    ppo = PolicyValueModel.from_checkpoint(
        _path(settings, settings.paths.ppo_checkpoint), device=args.device
    )
    reward = RewardModel.from_checkpoint(
        _path(settings, settings.paths.reward_checkpoint), device=args.device
    )
    result = evaluate_policies(
        prompts,
        sft,
        ppo,
        reward,
        OpenAIJudge(settings.evaluation.judge_model),
        max_new_tokens=settings.models.max_new_tokens,
    )
    output = Path(args.output) if args.output else _path(settings, settings.paths.evaluation_figure)
    plot_evaluation(result, output)
    result.save(output.with_suffix(".json"))
    logging.info("PPO win rate: %.2f%%", 100 * result.ppo_win_rate)
