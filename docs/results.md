# Results and interpretation

## Historical evaluation snapshot

The committed evaluation artifact reports:

| Policy | Automated judge wins | Share of valid decisions |
|---|---:|---:|
| SFT GPT-2 | 53 | 8.8% |
| PPO-aligned GPT-2 | 546 | 91.2% |
| **Total** | **599** | **100%** |

![Historical evaluation](assets/evaluation.png)

The second panel plots the absolute difference between the RoBERTa reward score and an LLM judge's
1–5 quality rating. Because the learned reward is an uncalibrated ranking score, this plot is a rough
agreement diagnostic—not a properly calibrated error metric.

## What the result supports

Within this particular prompt set, generation setup, and automated judge, the PPO checkpoint was
preferred substantially more often than the SFT checkpoint. This is evidence that the optimization
changed behavior in the direction measured by the judge.

## What the result does not establish

The result does not prove general helpfulness, factuality, harmlessness, or a 91.2% human preference
rate. Important sources of uncertainty include:

- evaluation prompts may overlap in style or content with training data;
- answer order and judge-model bias may affect preferences;
- one judge model is not an independent ground truth;
- stochastic decoding and absent confidence intervals reduce repeatability;
- reward scores and 1–5 ratings are on different scales;
- malformed or skipped judge responses must be reported explicitly.

## Recommended next evaluation

1. Freeze a versioned prompt set with train/test deduplication.
2. Run deterministic decoding and repeat sampled decoding across multiple seeds.
3. Randomize and swap answer order to measure position bias.
4. Add length, toxicity, repetition, and factuality diagnostics.
5. Collect blinded human pairwise preferences with inter-rater agreement.
6. Report bootstrap confidence intervals and all invalid/skipped judgments.
