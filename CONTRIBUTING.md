# Contributing

Contributions that improve correctness, reproducibility, evaluation quality, or documentation are
welcome.

## Development setup

```bash
git clone https://github.com/Dochikhoa2006/Reinforcement-Learning-Human-Feedback-Chatbot.git
cd Reinforcement-Learning-Human-Feedback-Chatbot
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[all,dev]"
```

Before opening a pull request, run:

```bash
python -m ruff check .
python -m ruff format --check .
python -m pytest
```

## Research changes

Changes to data processing, prompts, metrics, optimization, or checkpoints should document:

- the hypothesis and expected effect;
- the exact configuration and random seed;
- the dataset version and sample count;
- before-and-after metrics, including unsuccessful results;
- hardware assumptions and material compute cost.

Never commit API keys, private preference data, or model artifacts outside Git LFS.
