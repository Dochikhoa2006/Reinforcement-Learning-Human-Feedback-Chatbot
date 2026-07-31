# Methodology

## 1. Supervised fine-tuning

For each prompt \(x\), the preferred response \(y^+\) is formatted as a causal language-model
sequence. Prompt and padding positions are masked so the loss supervises only response tokens:

\[
\mathcal{L}_{SFT} = -\sum_t \log \pi_\theta(y_t^+ \mid x, y_{<t}^+)
\]

This stage establishes the initial instruction-conditioned policy and initializes the PPO actor.

## 2. Pairwise reward modeling

RoBERTa encodes the prompt with either the preferred response \(y^+\) or rejected response \(y^-\).
Masked mean pooling and a scalar head produce \(r_\phi(x,y)\). Training minimizes margin-ranking
loss:

\[
\mathcal{L}_{RM} = \max(0, m - r_\phi(x,y^+) + r_\phi(x,y^-))
\]

The score is ordinal, not automatically calibrated to a human rating scale.

## 3. PPO alignment

The trainable policy samples a response. A frozen SFT reference policy supplies a token-level drift
penalty:

\[
r_t^{KL} = -\beta\left(\log \pi_\theta(a_t\mid s_t) -
\log \pi_{ref}(a_t\mid s_t)\right)
\]

The reward-model score is added at the final response token. Returns and value estimates form the
advantage \(A_t\). PPO optimizes the clipped surrogate:

\[
\mathcal{L}_{policy} = -\mathbb{E}_t\left[\min(\rho_t A_t,
\operatorname{clip}(\rho_t,1-\epsilon,1+\epsilon)A_t)\right]
\]

where \(\rho_t = \pi_\theta(a_t\mid s_t) / \pi_{old}(a_t\mid s_t)\). A mean-squared value loss and
gradient clipping stabilize each update.

## Reproducibility controls

- The default random seed and all material hyperparameters live in `configs/default.toml`.
- Commands accept sample limits for inexpensive smoke experiments.
- Checkpoints identify SFT, reward-model, and PPO stages independently.
- Evaluation emits JSON in addition to the presentation figure.
- Unit tests cover dataset adapters, configuration, returns, and the clipped PPO objective.

## Known methodological limitations

- UltraFeedback preferences are synthetic/model-generated rather than newly collected from users.
- The implementation is intentionally compact and does not include GAE, reward whitening across
  batches, adaptive KL control, distributed rollout workers, or checkpoint resumability.
- The reward model can be exploited and is not a substitute for human evaluation.
- GPT-2's capacity and context window limit response quality and instruction following.
- The historical evaluation uses an LLM judge and should be validated with blinded human ratings.
