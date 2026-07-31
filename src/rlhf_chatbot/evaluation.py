"""Reproducible policy comparison with a pluggable judge."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from rlhf_chatbot.models.policy import PolicyValueModel
from rlhf_chatbot.models.reward import RewardModel


class Judge(Protocol):
    def prefer(self, prompt: str, answer_a: str, answer_b: str) -> str: ...

    def quality_score(self, prompt: str, answer: str) -> float: ...


class OpenAIJudge:
    """Small deterministic adapter around an OpenAI chat-completions model."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        from openai import OpenAI

        self.model = model
        self.client = OpenAI()

    def _complete(self, system: str, user: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        return (response.choices[0].message.content or "").strip()

    def prefer(self, prompt: str, answer_a: str, answer_b: str) -> str:
        result = self._complete(
            "You are an impartial response-quality evaluator.",
            (
                f"Question:\n{prompt}\n\nAnswer A:\n{answer_a}\n\n"
                f"Answer B:\n{answer_b}\n\nChoose the better answer. "
                "Respond with exactly A or B."
            ),
        ).upper()
        if result not in {"A", "B"}:
            raise ValueError(f"Judge returned an invalid preference: {result!r}")
        return result

    def quality_score(self, prompt: str, answer: str) -> float:
        result = self._complete(
            "You are a strict response-quality evaluator.",
            (
                f"Question:\n{prompt}\n\nAnswer:\n{answer}\n\n"
                "Score the answer from 1 to 5. Respond with only a number."
            ),
        )
        score = float(result)
        if not 1.0 <= score <= 5.0:
            raise ValueError(f"Judge returned an out-of-range score: {score}")
        return score


@dataclass
class EvaluationResult:
    sft_wins: int
    ppo_wins: int
    reward_judge_absolute_errors: list[float]
    skipped_preferences: int = 0
    skipped_scores: int = 0

    @property
    def evaluated_preferences(self) -> int:
        return self.sft_wins + self.ppo_wins

    @property
    def ppo_win_rate(self) -> float:
        if not self.evaluated_preferences:
            return 0.0
        return self.ppo_wins / self.evaluated_preferences

    def save(self, path: str | Path) -> None:
        payload = asdict(self) | {
            "evaluated_preferences": self.evaluated_preferences,
            "ppo_win_rate": self.ppo_win_rate,
        }
        Path(path).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def evaluate_policies(
    prompts: list[str],
    sft_policy: PolicyValueModel,
    ppo_policy: PolicyValueModel,
    reward_model: RewardModel,
    judge: Judge,
    *,
    max_new_tokens: int = 128,
) -> EvaluationResult:
    """Compare SFT and PPO policies and retain malformed-judge counts."""

    result = EvaluationResult(0, 0, [])
    for prompt in prompts:
        sft_answer = sft_policy.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        ppo_answer = ppo_policy.generate(prompt, max_new_tokens=max_new_tokens, do_sample=False)
        try:
            preferred = judge.prefer(prompt, sft_answer, ppo_answer)
        except (TypeError, ValueError):
            result.skipped_preferences += 1
            continue

        if preferred == "A":
            result.sft_wins += 1
        else:
            result.ppo_wins += 1

        try:
            learned_reward = reward_model.score(prompt, ppo_answer)
            judge_score = judge.quality_score(prompt, ppo_answer)
            result.reward_judge_absolute_errors.append(abs(judge_score - learned_reward))
        except (TypeError, ValueError):
            result.skipped_scores += 1
    return result


def plot_evaluation(result: EvaluationResult, output_path: str | Path) -> None:
    """Create the portfolio evaluation figure without opening an interactive window."""

    import matplotlib.pyplot as plt

    figure, (wins_axis, error_axis) = plt.subplots(1, 2, figsize=(15, 7))
    names = ["SFT (GPT-2)", "PPO-aligned GPT-2"]
    values = [result.sft_wins, result.ppo_wins]
    bars = wins_axis.bar(names, values, color=["#94A3B8", "#0EA5E9"])
    wins_axis.bar_label(bars, fontweight="bold")
    wins_axis.set_title("LLM-judge preference comparison")
    wins_axis.set_ylabel("Wins")

    error_axis.plot(result.reward_judge_absolute_errors, color="#0EA5E9", linewidth=1)
    error_axis.set_title("Reward-model vs. judge score difference")
    error_axis.set_xlabel("Evaluation sample")
    error_axis.set_ylabel("Absolute difference")
    error_axis.grid(alpha=0.25)

    figure.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)
