# Model checkpoints

The repository uses Git LFS for model weights.

- `sft/` — supervised GPT-2 policy and value head
- `reward/` — RoBERTa encoder and pairwise ranking head
- `ppo/` — final PPO-aligned GPT-2 policy and value head

Each directory is loadable through the corresponding class in `rlhf_chatbot.models`.
