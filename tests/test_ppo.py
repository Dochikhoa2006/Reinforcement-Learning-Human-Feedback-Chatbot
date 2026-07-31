import torch

from rlhf_chatbot.training.ppo import clipped_policy_loss, discounted_returns


def test_clipped_policy_loss_matches_unclipped_ratio_inside_range() -> None:
    old_log_probs = torch.log(torch.tensor([0.5, 0.5]))
    log_probs = torch.log(torch.tensor([0.55, 0.45]))
    advantages = torch.tensor([1.0, -1.0])

    loss = clipped_policy_loss(log_probs, old_log_probs, advantages, 0.2)

    expected = -torch.mean(torch.tensor([1.1, -0.9]))
    assert torch.isclose(loss, expected)


def test_discounted_returns_are_reverse_cumulative_rewards() -> None:
    rewards = torch.tensor([-0.1, -0.2, 1.0])

    returns = discounted_returns(rewards)

    assert torch.allclose(returns, torch.tensor([0.7, 0.8, 1.0]))
