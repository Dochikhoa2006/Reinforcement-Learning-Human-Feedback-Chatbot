"""Compact PPO stage with an explicit frozen reference policy."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor, nn

from rlhf_chatbot.config import Settings
from rlhf_chatbot.data import iter_parquet_preferences
from rlhf_chatbot.device import set_seed
from rlhf_chatbot.models.policy import PolicyValueModel
from rlhf_chatbot.models.reward import RewardModel

LOGGER = logging.getLogger(__name__)


def clipped_policy_loss(
    log_probs: Tensor,
    old_log_probs: Tensor,
    advantages: Tensor,
    clip_epsilon: float,
) -> Tensor:
    """Compute the clipped PPO surrogate loss."""

    ratio = torch.exp(log_probs - old_log_probs)
    clipped_ratio = ratio.clamp(1.0 - clip_epsilon, 1.0 + clip_epsilon)
    return -torch.minimum(ratio * advantages, clipped_ratio * advantages).mean()


def token_statistics(model: PolicyValueModel, input_ids: Tensor) -> tuple[Tensor, Tensor]:
    """Return next-token log probabilities and aligned value predictions."""

    logits, values = model(input_ids, torch.ones_like(input_ids))
    targets = input_ids[:, 1:].unsqueeze(-1)
    log_probs = torch.log_softmax(logits[:, :-1], dim=-1).gather(-1, targets).squeeze(-1)
    return log_probs, values[:, :-1]


def discounted_returns(token_rewards: Tensor) -> Tensor:
    """Undiscounted return-to-go for a short generated response."""

    return torch.flip(torch.cumsum(torch.flip(token_rewards, dims=(0,)), dim=0), dims=(0,))


def train_ppo(
    settings: Settings,
    *,
    dataset_path: str | Path,
    sft_checkpoint: str | Path,
    reward_checkpoint: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    device: str = "auto",
) -> PolicyValueModel:
    """Optimize an SFT policy against the learned reward while controlling KL drift."""

    set_seed(settings.seed)
    policy = PolicyValueModel.from_checkpoint(sft_checkpoint, device=device)
    reference = PolicyValueModel.from_checkpoint(sft_checkpoint, device=device)
    reward_model = RewardModel.from_checkpoint(reward_checkpoint, device=device)
    reference.eval().requires_grad_(False)
    reward_model.eval().requires_grad_(False)

    optimizer = torch.optim.AdamW(policy.parameters(), lr=settings.ppo.learning_rate)
    value_objective = nn.MSELoss()

    for step, example in enumerate(iter_parquet_preferences(dataset_path, limit=limit), start=1):
        policy.eval()
        full_ids, generated_ids = policy.generate_token_ids(
            example.prompt,
            max_new_tokens=settings.models.max_new_tokens,
            do_sample=True,
        )
        if generated_ids.numel() == 0:
            LOGGER.warning("ppo step=%d skipped empty generation", step)
            continue

        prompt_length = full_ids.shape[1] - generated_ids.shape[1]
        answer = policy.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()
        terminal_reward = reward_model.score(
            example.prompt, answer, max_length=settings.models.max_length
        )

        with torch.no_grad():
            old_log_probs, old_values = token_statistics(policy, full_ids)
            reference_log_probs, _ = token_statistics(reference, full_ids)
            response_slice = slice(prompt_length - 1, None)
            old_response_log_probs = old_log_probs[0, response_slice]
            reference_response_log_probs = reference_log_probs[0, response_slice]
            token_rewards = -settings.ppo.kl_coefficient * (
                old_response_log_probs - reference_response_log_probs
            )
            token_rewards[-1] += terminal_reward
            returns = discounted_returns(token_rewards)
            advantages = returns - old_values[0, response_slice]
            if advantages.numel() > 1:
                advantages = (advantages - advantages.mean()) / (
                    advantages.std(unbiased=False) + 1e-8
                )

        policy.train()
        for _ in range(settings.ppo.update_epochs):
            current_log_probs, current_values = token_statistics(policy, full_ids)
            current_log_probs = current_log_probs[0, response_slice]
            current_values = current_values[0, response_slice]
            policy_loss = clipped_policy_loss(
                current_log_probs,
                old_response_log_probs,
                advantages,
                settings.ppo.clip_epsilon,
            )
            value_loss = value_objective(current_values, returns)
            loss = policy_loss + settings.ppo.value_coefficient * value_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(policy.parameters(), settings.ppo.max_grad_norm)
            optimizer.step()

        LOGGER.info("ppo step=%d reward=%.4f loss=%.4f", step, terminal_reward, loss.item())

    policy.save_checkpoint(output_dir)
    return policy
