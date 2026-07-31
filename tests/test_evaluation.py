from pathlib import Path

from rlhf_chatbot.evaluation import EvaluationResult, evaluate_policies


class FakePolicy:
    def __init__(self, answer: str) -> None:
        self.answer = answer

    def generate(self, prompt: str, **_: object) -> str:
        return f"{self.answer}: {prompt}"


class FakeReward:
    def score(self, prompt: str, answer: str) -> float:
        return 3.0


class FakeJudge:
    def prefer(self, prompt: str, answer_a: str, answer_b: str) -> str:
        return "B"

    def quality_score(self, prompt: str, answer: str) -> float:
        return 4.0


def test_policy_evaluation_tracks_wins_and_score_differences() -> None:
    result = evaluate_policies(
        ["Question"],
        FakePolicy("SFT"),  # type: ignore[arg-type]
        FakePolicy("PPO"),  # type: ignore[arg-type]
        FakeReward(),  # type: ignore[arg-type]
        FakeJudge(),
    )

    assert result.ppo_wins == 1
    assert result.ppo_win_rate == 1.0
    assert result.reward_judge_absolute_errors == [1.0]


def test_evaluation_result_serializes_derived_metrics(tmp_path: Path) -> None:
    result = EvaluationResult(sft_wins=1, ppo_wins=3, reward_judge_absolute_errors=[0.5])
    output = tmp_path / "result.json"

    result.save(output)

    assert '"ppo_win_rate": 0.75' in output.read_text(encoding="utf-8")
