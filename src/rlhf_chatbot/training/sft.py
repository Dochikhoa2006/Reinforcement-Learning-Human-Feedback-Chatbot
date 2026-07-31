"""Supervised fine-tuning stage."""

from __future__ import annotations

import logging
from pathlib import Path

import torch
from torch import Tensor, nn

from rlhf_chatbot.config import Settings
from rlhf_chatbot.data import PreferenceExample, batched, iter_parquet_preferences
from rlhf_chatbot.device import set_seed
from rlhf_chatbot.models.policy import PolicyValueModel

LOGGER = logging.getLogger(__name__)


def encode_sft_batch(
    model: PolicyValueModel,
    examples: list[PreferenceExample],
    max_length: int,
) -> tuple[Tensor, Tensor, Tensor]:
    """Encode examples and mask prompt/padding tokens from the language loss."""

    rows: list[list[int]] = []
    labels: list[list[int]] = []
    for example in examples:
        prompt_ids = model.tokenizer.encode(
            model.format_prompt(example.prompt), add_special_tokens=False
        )
        response_text = (
            f" [assistant]: {example.chosen} {model.tokenizer.eos_token} "
            f"{model.tokenizer.sep_token}"
        )
        response_ids = model.tokenizer.encode(response_text, add_special_tokens=False)

        # Preserve response supervision when unusually long prompts are encountered.
        response_ids = response_ids[: max(1, max_length // 2)]
        prompt_ids = prompt_ids[: max_length - len(response_ids)]
        token_ids = prompt_ids + response_ids
        rows.append(token_ids)
        labels.append([-100] * len(prompt_ids) + response_ids)

    width = max(len(row) for row in rows)
    pad_id = model.tokenizer.pad_token_id
    padded_rows = [row + [pad_id] * (width - len(row)) for row in rows]
    padded_labels = [row + [-100] * (width - len(row)) for row in labels]
    attention = [[1] * len(row) + [0] * (width - len(row)) for row in rows]
    device = model.runtime_device
    return (
        torch.tensor(padded_rows, dtype=torch.long, device=device),
        torch.tensor(attention, dtype=torch.long, device=device),
        torch.tensor(padded_labels, dtype=torch.long, device=device),
    )


def train_sft(
    settings: Settings,
    *,
    dataset_path: str | Path,
    output_dir: str | Path,
    limit: int | None = None,
    device: str = "auto",
) -> PolicyValueModel:
    """Train the policy on preferred answers and persist a compatible checkpoint."""

    set_seed(settings.seed)
    model = PolicyValueModel.from_base_model(settings.models.policy_base, device=device)
    parameters = list(model.backbone.parameters()) + list(model.policy_head.parameters())
    optimizer = torch.optim.AdamW(parameters, lr=settings.sft.learning_rate)
    loss_function = nn.CrossEntropyLoss(ignore_index=-100)

    model.train()
    for epoch in range(settings.sft.epochs):
        records = iter_parquet_preferences(dataset_path, limit=limit)
        for step, examples in enumerate(batched(records, settings.sft.batch_size), start=1):
            input_ids, attention_mask, labels = encode_sft_batch(
                model, examples, settings.models.max_length
            )
            logits, _ = model(input_ids, attention_mask)
            loss = loss_function(
                logits[:, :-1].reshape(-1, logits.shape[-1]),
                labels[:, 1:].reshape(-1),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            LOGGER.info("sft epoch=%d step=%d loss=%.4f", epoch + 1, step, loss.item())

    model.save_checkpoint(output_dir)
    return model
