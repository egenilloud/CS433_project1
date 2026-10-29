# Ablation feature engineering — Newton, 5-fold CV stratifiée (seed 1)

Seuil choisi sur les scores hors fold (OOF) ; F1 de chaque fold à ce seuil commun. Le gain est apparié (mêmes folds que la baseline).

### 1. Ajouts cumulatifs

| Configuration | λ | # features | F1 CV (moy ± std) | gain vs baseline (folds améliorés) | seuil |
|---|---|---|---|---|---|
| baseline (preprocessing d'origine) | 0.0001 | 119 | 0.4241 ± 0.0058 | — | -1.39 |
| + âge one-hot (_AGEG5YR) | 0.0001 | 133 | 0.4243 ± 0.0057 | +0.0002 (2/5) | -1.39 |
| + âge² et IMC² | 0.0001 | 135 | 0.4243 ± 0.0061 | +0.0002 (3/5) | -1.37 |
| + IMC one-hot (_BMI5CAT) | 0.0001 | 140 | 0.4242 ± 0.0063 | +0.0001 (3/5) | -1.37 |
| + GENHLTH one-hot | 0.0001 | 146 | 0.4243 ± 0.0058 | +0.0002 (2/5) | -1.37 |
| + 12 interactions ciblées | 0.0001 | 158 | 0.4257 ± 0.0052 | +0.0016 (4/5) | -1.35 |
| + modules optionnels (aspirine, insuline...) | 0.0001 | 171 | 0.4268 ± 0.0053 | +0.0027 (4/5) | -1.36 |
| + 105 interactions (15 facteurs de risque) | 0.0001 | 264 | 0.4315 ± 0.0048 | +0.0074 (5/5) | -1.36 |
| + _STATE one-hot | 0.0001 | 318 | 0.4312 ± 0.0048 | +0.0071 (5/5) | -1.38 |

### 2. Retrait d'une option (depuis la config complète)

| Configuration | λ | # features | F1 CV (moy ± std) | gain vs baseline (folds améliorés) | seuil |
|---|---|---|---|---|---|
| + _STATE one-hot | 0.0001 | 318 | 0.4312 ± 0.0048 | +0.0071 (5/5) | -1.38 |
| complet sans age_onehot | 0.0001 | 304 | 0.4315 ± 0.0045 | +0.0074 (5/5) | -1.34 |
| complet sans squares | 0.0001 | 316 | 0.4310 ± 0.0056 | +0.0069 (5/5) | -1.32 |
| complet sans bmi_onehot | 0.0001 | 313 | 0.4311 ± 0.0049 | +0.0070 (5/5) | -1.38 |
| complet sans genhlth_onehot | 0.0001 | 312 | 0.4311 ± 0.0053 | +0.0069 (5/5) | -1.35 |
| complet sans interactions | 0.0001 | 318 | 0.4312 ± 0.0048 | +0.0071 (5/5) | -1.38 |
| complet sans extra_modules | 0.0001 | 305 | 0.4297 ± 0.0052 | +0.0055 (5/5) | -1.38 |
| complet sans pairwise_top | 0.0001 | 225 | 0.4284 ± 0.0051 | +0.0043 (5/5) | -1.32 |
| complet sans _STATE | 0.0001 | 264 | 0.4315 ± 0.0048 | +0.0074 (5/5) | -1.36 |

### 3. Choix de λ (config complète)

| Configuration | λ | # features | F1 CV (moy ± std) | gain vs baseline (folds améliorés) | seuil |
|---|---|---|---|---|---|
| complet | 1e-06 | 318 | 0.4312 ± 0.0055 | +0.0071 (5/5) | -1.36 |
| complet | 1e-05 | 318 | 0.4311 ± 0.0053 | +0.0070 (5/5) | -1.33 |
| complet | 0.0001 | 318 | 0.4312 ± 0.0048 | +0.0071 (5/5) | -1.38 |
| complet | 0.001 | 318 | 0.4285 ± 0.0049 | +0.0044 (5/5) | -1.41 |
| complet | 0.01 | 318 | 0.4228 ± 0.0049 | -0.0013 (1/5) | -1.61 |

Meilleur λ : 1e-06 (F1 CV = 0.4312) ; F1 stable pour λ <= 1e-4, chute au-delà.

### 4. Config retenue vs baseline

| Configuration | λ | # features | F1 CV (moy ± std) | gain vs baseline (folds améliorés) | seuil |
|---|---|---|---|---|---|
| baseline (preprocessing d'origine) | 0.0001 | 119 | 0.4241 ± 0.0058 | — | -1.39 |
| config retenue (RECOMMENDED_FEATURES) | 0.0001 | 264 | 0.4315 ± 0.0048 | +0.0074 (5/5) | -1.36 |
