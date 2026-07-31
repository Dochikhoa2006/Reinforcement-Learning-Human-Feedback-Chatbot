# GitHub and LinkedIn publishing checklist

## Before publishing

- [ ] Review `git diff` and confirm no secrets or private data are present.
- [ ] Run `python -m ruff check .`, `python -m ruff format --check .`, and `python -m pytest`.
- [ ] Confirm Git LFS is installed and run `git lfs status`.
- [ ] Confirm all three checkpoint directories contain their heads, tokenizer, config, and backbone.
- [ ] Open the README on GitHub and verify Mermaid, equations, badges, and relative links.
- [ ] Build and run the Docker image with Docker Desktop active.
- [ ] Test the app on one CPU environment in addition to the development accelerator.

## Recommended GitHub About settings

**Description**

> Educational end-to-end RLHF pipeline with GPT-2, RoBERTa reward modeling, PPO, Streamlit, tests,
> and reproducible ML tooling.

**Website**

Use the deployed Streamlit URL when one is available; otherwise leave this blank rather than linking
back to the repository.

**Topics**

`rlhf`, `llm-alignment`, `reinforcement-learning`, `ppo`, `reward-modeling`, `gpt2`, `roberta`,
`pytorch`, `streamlit`, `mlops`

Keep the topic list focused. Specific optimizer and loss-function names make discovery noisier and
are already documented in the README.

## Recommended repository settings

- Enable **Issues** and **Discussions** for feedback and research questions.
- Enable **Actions** and require the `quality` CI job before merging to `main`.
- Enable **Dependabot alerts** and private vulnerability reporting.
- Use squash merging for a readable research history.
- Add a social preview based on the architecture and evaluation, with the title readable at mobile
  size.
- Pin the repository on the GitHub profile after the `v1.0.0` release is published.

## Suggested first release

Tag: `v1.0.0`

Title: `Portfolio-ready RLHF pipeline`

Release notes:

> Reorganizes the original research scripts into an installable, tested RLHF package. The release
> includes stage-specific CLIs, typed configuration, migrated Git LFS checkpoints, a lightweight
> Streamlit app, Docker runtime, automated CI, methodology/results documentation, and a model card.

Attach no duplicate weight archives—the release should use the Git LFS files already versioned in
the repository.

## LinkedIn launch sequence

1. Publish the GitHub release and verify the public README while signed out.
2. Record a short, captioned Streamlit demonstration.
3. Use the copy in [linkedin-post.md](linkedin-post.md), retaining the evaluation caveat.
4. Lead the carousel with the architecture, then the result, repository structure, and demo.
5. Reply to technical questions with links to the methodology and results pages.

## Short profile blurb

> Built an end-to-end RLHF research pipeline that aligns GPT-2 through supervised fine-tuning,
> RoBERTa preference modeling, and PPO with a frozen KL reference. Packaged it with reproducible
> configuration, CI, tests, Docker, Streamlit, Git LFS checkpoints, and transparent evaluation
> limitations.
