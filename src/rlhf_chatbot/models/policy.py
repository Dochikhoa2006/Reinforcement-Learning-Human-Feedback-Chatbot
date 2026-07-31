"""GPT-2 policy with a scalar value head."""

from __future__ import annotations

from pathlib import Path

import torch
from torch import Tensor, nn
from transformers import GPT2Model, GPT2Tokenizer

from rlhf_chatbot.device import resolve_device

SPECIAL_TOKENS = {
    "pad_token": "[PAD]",
    "cls_token": "[CLS]",
    "sep_token": "[SEP]",
    "bos_token": "[BOS]",
}


def _load_weights(path: Path, device: torch.device) -> dict[str, Tensor]:
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:  # PyTorch < 2.0 compatibility
        return torch.load(path, map_location=device)


class PolicyValueModel(nn.Module):
    """GPT-2 backbone with independent language-policy and value heads."""

    def __init__(self, backbone: GPT2Model, tokenizer: GPT2Tokenizer) -> None:
        super().__init__()
        self.backbone = backbone
        self.tokenizer = tokenizer
        self.policy_head = nn.Linear(backbone.config.n_embd, len(tokenizer))
        self.value_head = nn.Linear(backbone.config.n_embd, 1)
        nn.init.normal_(self.value_head.weight, mean=0.0, std=0.001)
        nn.init.zeros_(self.value_head.bias)

    @classmethod
    def from_base_model(
        cls,
        model_name: str = "gpt2",
        *,
        device: str = "auto",
    ) -> PolicyValueModel:
        tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        backbone = GPT2Model.from_pretrained(model_name)
        added = tokenizer.add_special_tokens(SPECIAL_TOKENS)
        if added:
            original_size = backbone.get_input_embeddings().weight.shape[0]
            backbone.resize_token_embeddings(len(tokenizer))
            with torch.no_grad():
                embeddings = backbone.get_input_embeddings().weight
                embeddings[original_size:] = embeddings[:original_size].mean(dim=0)
        tokenizer.padding_side = "left"
        return cls(backbone, tokenizer).to(resolve_device(device))

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: str = "auto",
    ) -> PolicyValueModel:
        checkpoint = Path(checkpoint)
        runtime_device = resolve_device(device)
        tokenizer = GPT2Tokenizer.from_pretrained(checkpoint)
        backbone = GPT2Model.from_pretrained(checkpoint)
        tokenizer.padding_side = "left"
        model = cls(backbone, tokenizer)
        model.policy_head.load_state_dict(
            _load_weights(checkpoint / "policy_head.pt", runtime_device)
        )
        model.value_head.load_state_dict(
            _load_weights(checkpoint / "value_head.pt", runtime_device)
        )
        return model.to(runtime_device)

    @property
    def runtime_device(self) -> torch.device:
        return next(self.parameters()).device

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> tuple[Tensor, Tensor]:
        hidden = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        ).last_hidden_state
        return self.policy_head(hidden), self.value_head(hidden).squeeze(-1)

    def format_prompt(self, prompt: str) -> str:
        return (
            f"{self.tokenizer.cls_token} {self.tokenizer.bos_token} "
            f"{prompt.strip()} {self.tokenizer.sep_token} {self.tokenizer.bos_token}"
        )

    def encode_prompt(self, prompt: str) -> Tensor:
        token_ids = self.tokenizer.encode(
            self.format_prompt(prompt),
            add_special_tokens=False,
        )
        return torch.tensor(token_ids, dtype=torch.long, device=self.runtime_device).unsqueeze(0)

    @torch.inference_mode()
    def generate_token_ids(
        self,
        prompt: str,
        *,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int = 50,
        do_sample: bool = True,
    ) -> tuple[Tensor, Tensor]:
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        if do_sample and top_k < 1:
            raise ValueError("top_k must be at least 1 when sampling")

        sequence = self.encode_prompt(prompt)
        generated: list[Tensor] = []
        context_limit = int(self.backbone.config.n_positions)
        stop_ids = {self.tokenizer.eos_token_id, self.tokenizer.sep_token_id}

        self.eval()
        for _ in range(max_new_tokens):
            if sequence.shape[1] >= context_limit:
                break
            logits, _ = self(sequence, torch.ones_like(sequence))
            next_logits = logits[:, -1, :] / temperature
            if do_sample:
                k = min(top_k, next_logits.shape[-1])
                threshold = torch.topk(next_logits, k).values[:, -1].unsqueeze(-1)
                next_logits = next_logits.masked_fill(next_logits < threshold, -torch.inf)
                token = torch.multinomial(torch.softmax(next_logits, dim=-1), 1)
            else:
                token = next_logits.argmax(dim=-1, keepdim=True)

            generated.append(token)
            sequence = torch.cat((sequence, token), dim=1)
            if token.item() in stop_ids:
                break

        generated_ids = (
            torch.cat(generated, dim=1)
            if generated
            else torch.empty((1, 0), dtype=torch.long, device=self.runtime_device)
        )
        return sequence, generated_ids

    def generate(self, prompt: str, **generation_kwargs: object) -> str:
        _, generated_ids = self.generate_token_ids(prompt, **generation_kwargs)
        if generated_ids.numel() == 0:
            return ""
        return self.tokenizer.decode(generated_ids[0], skip_special_tokens=True).strip()

    def save_checkpoint(self, output_dir: str | Path) -> None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self.backbone.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        torch.save(self.policy_head.state_dict(), output_dir / "policy_head.pt")
        torch.save(self.value_head.state_dict(), output_dir / "value_head.pt")
