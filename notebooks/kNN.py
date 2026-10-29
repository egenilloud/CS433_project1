# functions for the k Nearest Neighbor model


import numpy as np

def knn_predict_eucl(X_train, y_train, X_test, k):
    """
    Predict binary labels using K-Nearest Neighbors.

    X_train: (N_train, n_features)
    y_train: (N_train,)
    X_test:  (N_test, n_features)
    k: number of neighbors
    """
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_test = np.asarray(X_test)

    if not 1 <= k <= len(X_train):
        raise ValueError("k must be between 1 and len(X_train)")

    y_pred = []

    for x in X_test:
        # 1. Compute Euclidean distances
        distances = np.sqrt(np.sum((X_train - x)**2, axis=1))

        # 2. Find the k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]

        # 3. Get their labels
        nearest_labels = y_train[nearest_indices]

        # 4. Majority vote (labels 0 and 1)
        prediction = int(np.mean(nearest_labels) > 0.5)

        y_pred.append(prediction)

    return np.array(y_pred)


import numpy as np

def knn_predict_manhattan(X_train, y_train, X_test, k):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_test = np.asarray(X_test)

    if not 1 <= k <= len(X_train):
        raise ValueError("Invalid k")

    y_pred = []

    for x in X_test:
        # 1. Manhattan distances
        distances = np.sum(np.abs(X_train - x), axis=1)

        # 2. Find k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]

        # 3. Get their labels
        nearest_labels = y_train[nearest_indices]

        # 4. Majority vote
        prediction = int(np.mean(nearest_labels) > 0.5)
        y_pred.append(prediction)

    return np.array(y_pred)


def knn_predict_minkowski(X_train, y_train, X_test, k, q=3):
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train).ravel()
    X_test = np.asarray(X_test)

    if not 1 <= k <= len(X_train):
        raise ValueError("Invalid k")
    if q < 1:
        raise ValueError("q must be >= 1")

    y_pred = []

    for x in X_test:
        # 1. Minkowski distances
        distances = np.sum(
            np.abs(X_train - x)**q, axis=1
        )**(1/q)

        # 2. Find k nearest neighbors
        nearest_indices = np.argsort(distances)[:k]

        # 3. Get their labels
        nearest_labels = y_train[nearest_indices]

        # 4. Majority vote
        prediction = int(np.mean(nearest_labels) > 0.5)
        y_pred.append(prediction)

    return np.array(y_pred)


# cross validation for K param --> you can change the method you're calling for distance calculation

def knn_cross_validation_f1(X, y, k_values, n_folds=5, seed=42):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y).ravel()

    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y))
    folds = np.array_split(indices, n_folds)

    mean_scores = []

    for k in k_values:
        fold_scores = []

        for i in range(n_folds):
            val_idx = folds[i]
            train_idx = np.concatenate([
                folds[j] for j in range(n_folds) if j != i
            ])

            X_train, y_train = X[train_idx], y[train_idx]
            X_val, y_val = X[val_idx], y[val_idx]

            # Standardize using training fold only
            mean = np.mean(X_train, axis=0)
            std = np.std(X_train, axis=0)
            std[std == 0] = 1

            X_train = (X_train - mean) / std
            X_val = (X_val - mean) / std

            # Predict using your KNN function
            y_pred = knn_predict_eucl(X_train, y_train, X_val, k)

            # Calculate F1-score
            score = f1_score(y_val, y_pred)
            fold_scores.append(score)

        mean_scores.append(np.mean(fold_scores))

    return np.array(mean_scores)


def f1_score(y_true, y_pred):
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # True positives, false positives, false negatives
    TP = np.sum((y_true == 1) & (y_pred == 1))
    FP = np.sum((y_true == 0) & (y_pred == 1))
    FN = np.sum((y_true == 1) & (y_pred == 0))

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)