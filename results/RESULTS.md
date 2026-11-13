| System | Accuracy % (95% CI) | Macro-F1 % (95% CI) | Invalid % | Prompt tokens | Examples/s | Peak mem GB |
|---|---|---|---|---|---|---|
| zero-shot | 19.1 [17.7, 20.5] | 19.4 [17.9, 20.5] | 39.5 | 448 | 3.61 | 2.18 |
| few-shot-8 | 76.2 [74.8, 77.7] | 78.8 [77.2, 79.9] | 6.7 | 651 | 2.39 | 3.25 |
| lora | 92.9 [92.0, 93.8] | 92.9 [92.0, 93.8] | 0.2 | 62 | 13.23 | 2.39 |

n = 3080 examples (test split); CIs: 1,000 bootstrap resamples.

Paired bootstrap, accuracy difference vs `few-shot-8`:

| System | Δ accuracy (pp) | 95% CI |
|---|---|---|
| zero-shot | -57.1 | [-59.0, -55.3] |
| lora | +16.6 | [+15.2, +18.1] |
