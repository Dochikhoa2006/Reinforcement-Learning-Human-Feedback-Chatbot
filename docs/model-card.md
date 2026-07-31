# Model card: PPO-aligned GPT-2 policy

## Model summary

The final model is a GPT-2 policy initialized through supervised fine-tuning and subsequently
optimized with PPO against a learned RoBERTa reward model. It is distributed as an educational
research artifact.

| Property | Value |
|---|---|
| Policy backbone | GPT-2 (12 layers, 768 hidden size) |
| Reward backbone | RoBERTa-base |
| Alignment stages | SFT → pairwise reward modeling → PPO |
| Training signal | UltraFeedback chosen/rejected response pairs |
| Interface | Python API and Streamlit app |
| License | CC BY 4.0 |

## Intended use

- studying the components of a compact RLHF pipeline;
- experimenting with reward modeling and PPO objectives;
- demonstrating ML engineering structure in a portfolio;
- local, low-stakes exploration of an aligned small language model.

## Out-of-scope use

Do not use the model for high-impact decisions, safety-critical automation, unsupervised public
deployment, impersonation, or generation of authoritative medical, legal, or financial advice.

## Training data

The pipeline expects the UltraFeedback binarized preference format: a prompt with chosen and rejected
chat responses. Users reproducing training are responsible for reviewing the upstream dataset card,
licenses, language distribution, synthetic-data provenance, and sensitive-content limitations.

## Evaluation

The stored automated-judge snapshot preferred the PPO policy in 546 of 599 valid comparisons
(91.2%). This result is not a human preference estimate. See [results](results.md) for limitations and
the proposed validation plan.

## Risks and limitations

- GPT-2 can hallucinate, repeat text, lose instruction context, and produce unsafe content.
- Preference data and an automated reward model can encode cultural and annotator bias.
- PPO can exploit reward-model shortcuts instead of improving true response quality.
- The reward score is relative and uncalibrated.
- No red-team, multilingual, demographic fairness, or production-security evaluation is claimed.

## Mitigations

The demo displays a research-use notice, limits generation length, avoids presenting output as
authoritative, separates secrets from source control, and documents evaluation uncertainty. Any real
deployment would require content controls, monitoring, human escalation, abuse testing, and a much
stronger model/evaluation program.
