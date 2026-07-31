# Changelog

All notable changes to this project are documented here.

## [1.0.0] - 2026-07-31

### Added

- installable `rlhf_chatbot` source package;
- typed TOML configuration and stage-specific CLI commands;
- isolated SFT, reward-model, PPO, evaluation, and Streamlit runtimes;
- frozen-reference PPO objective and response-only SFT masking;
- unit tests, Ruff configuration, GitHub Actions, and Dependabot;
- Docker runtime, model card, methodology, results, and LinkedIn showcase copy.

### Changed

- moved checkpoints to `artifacts/models/` and documentation assets to `docs/`;
- moved local datasets under `data/` and excluded them from source control;
- removed import-time Spark startup from model and inference code;
- replaced flat research scripts with reusable modules and thin executable wrappers.
