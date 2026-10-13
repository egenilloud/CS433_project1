# functions for the k Nearest Neighbor model


import numpy as np

def knn_predict(X_train, y_train, X_test, k):
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
