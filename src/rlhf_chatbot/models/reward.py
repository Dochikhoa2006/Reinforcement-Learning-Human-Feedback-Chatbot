"""RoBERTa preference reward model."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import RobertaModel, RobertaTokenizer

from rlhf_chatbot.device import resolve_device
from rlhf_chatbot.models.policy import _load_weights


class RewardModel(nn.Module):
    """RoBERTa encoder with mean pooling and a scalar ranking head."""

    def __init__(self, backbone: RobertaModel, tokenizer: RobertaTokenizer) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.ranking_head = nn.Linear(backbone.config.hidden_size, 1)

    @classmethod
    def from_base_model(
        cls,
        model_name: str = "roberta-base",
        *,
        device: str = "auto",
    ) -> RewardModel:
        model = cls(
            RobertaModel.from_pretrained(model_name),
            RobertaTokenizer.from_pretrained(model_name),
        )
        return model.to(resolve_device(device))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "auto",
    ) -> RewardModel:
        checkpoint = Path(checkpoint)
        runtime_device = resolve_device(device)
        model = cls(
            RobertaModel.from_pretrained(checkpoint),
            RobertaTokenizer.from_pretrained(checkpoint),
        )
        model.ranking_head.load_state_dict(
            _load_weights(checkpoint / "pairwise_ranking_head.pt", runtime_device)
        )
        return model.to(runtime_device)

    @property
    def runtime_device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand_as(hidden).float()
        pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1e-9)
        return self.ranking_head(pooled).squeeze(-1)

    def encode_pairs(
        self,
        prompts: list[str],
        answers: list[str],
        *,
        max_length: int = 512,
    ) -> dict[str, Tensor]:
        encoded = self.tokenizer(
            prompts,
            answers,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        return {key: value.to(self.runtime_device) for key, value in encoded.items()}

    @torch.inference_mode()
    def score(self, prompt: str, answer: str, *, max_length: int = 512) -> float:
        self.eval()
        encoded = self.encode_pairs([prompt], [answer], max_length=max_length)
        return float(self(encoded["input_ids"], encoded["attention_mask"]).item())

    def save_checkpoint(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.ranking_head.state_dict(), output_dir / "pairwise_ranking_head.pt")
