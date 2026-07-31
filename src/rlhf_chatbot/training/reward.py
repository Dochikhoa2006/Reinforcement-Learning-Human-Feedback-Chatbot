"""Pairwise reward-model training stage."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import nn

from rlhf_chatbot.config import Settings
from rlhf_chatbot.data import batched, iter_parquet_preferences
from rlhf_chatbot.device import set_seed
from rlhf_chatbot.models.reward import RewardModel

LOGGER = logging.getLogger(__name__)


def train_reward_model(
    settings: Settings,
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    device: str = "auto",
) -> RewardModel:
    """Fit a scalar reward function with a margin-ranking objective."""

    set_seed(settings.seed)
    model = RewardModel.from_base_model(settings.models.reward_base, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.reward.learning_rate)
    objective = nn.MarginRankingLoss(margin=settings.reward.margin)

    model.train()
    for epoch in range(settings.reward.epochs):
        records = iter_parquet_preferences(dataset_path, limit=limit)
        for step, examples in enumerate(batched(records, settings.reward.batch_size), start=1):
            prompts = [example.prompt for example in examples]
            chosen = [example.chosen for example in examples]
            rejected = [example.rejected or "" for example in examples]
            chosen_inputs = model.encode_pairs(
                prompts, chosen, max_length=settings.models.max_length
            )
            rejected_inputs = model.encode_pairs(
                prompts, rejected, max_length=settings.models.max_length
            )
            chosen_scores = model(chosen_inputs["input_ids"], chosen_inputs["attention_mask"])
            rejected_scores = model(rejected_inputs["input_ids"], rejected_inputs["attention_mask"])
            targets = torch.ones_like(chosen_scores)
            loss = objective(chosen_scores, rejected_scores, targets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            LOGGER.info("reward epoch=%d step=%d loss=%.4f", epoch + 1, step, loss.item())

    model.save_checkpoint(output_dir)
    return model
