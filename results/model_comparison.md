# Model comparison — 5-fold stratified CV (seed 1), lambda = 0.0001

| Model | F1 CV (mean ± std) | Accuracy |
|---|---|---|
| Trivial: predict all positive | 0.1623 ± 0.0000 | 0.088 |
| Ridge regression (labels -1/1), tuned threshold | 0.4102 ± 0.0061 | 0.857 |
| Reg. logistic regression, GD (300 it.), tuned thr. | 0.4168 ± 0.0033 | 0.867 |
| Reg. logistic regression, Newton, threshold 0.5 | 0.2226 ± 0.0060 | 0.914 |
| Reg. logistic regression, Newton, tuned threshold | 0.4241 ± 0.0058 | 0.871 |
| + feature engineering (final model) | 0.4315 ± 0.0048 | 0.871 |
