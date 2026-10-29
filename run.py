
import numpy as np

import preprocessing as prep
import cross_validation as cv
from helpers import load_csv_data, create_csv_submission

DATA_PATH = "data/dataset"
OUTPUT = "submission.csv"

FEATURES = prep.RECOMMENDED_FEATURES  # feature engineering options kept after ablation
USE_STATE = prep.RECOMMENDED_USE_STATE  # _STATE one-hot (False: no gain in CV)
LAMBDA = prep.RECOMMENDED_LAMBDA  # L2 regularization strength (1e-4)
K_FOLDS = 5
SEED = 1


def main():
    np.random.seed(SEED)

    # 1. Load data
    print("Loading data...")
    x_train, x_test, y_train, _, test_ids = load_csv_data(DATA_PATH)
    names = prep.load_feature_names(f"{DATA_PATH}/x_train.csv")

    # 2-3. Cross-validation: estimate F1 and choose the decision threshold
    print(f"{K_FOLDS}-fold cross-validation...")
    res = cv.cv_newton(
        x_train, y_train, names,
        features=FEATURES, lambda_=LAMBDA, k=K_FOLDS, seed=SEED,
        use_state=USE_STATE, verbose=False,
    )
    print(f"CV F1 = {res['f1_mean']:.4f} +- {res['f1_std']:.4f} "
          f"({res['n_features']} features, threshold = {res['threshold']:.3f})")

    # 4. Final model on the whole training set, predictions on the test set
    print("Training final model and predicting...")
    pred, _, _ = cv.fit_final_newton(
        x_train, y_train, x_test, names, res["threshold"],
        features=FEATURES, lambda_=LAMBDA, use_state=USE_STATE,
    )
    create_csv_submission(test_ids, pred, OUTPUT)
    print(f"-> {OUTPUT} written ({np.mean(pred == 1):.1%} predicted positive)")


if __name__ == "__main__":
    main()
