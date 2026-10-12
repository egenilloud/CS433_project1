import numpy as np


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    "Linear regression using gradient descent"

    return NotImplementedError



def mean_squared_error_sgd(y, tx, initial w, max_iters, gamma):
    "Linear regression using stochastic gradient descent"

    return NotImplementedError



def least_squares(y, tx):
    "Least squares regression using normal equations"
    A = tx.T.dot(tx)
    b = tx.T.dot(y)
    w = np.linalg.solve(A, b)
    e = y - tx.dot(w)
    mse = 1 / 2 * np.mean(e**2)

    return mse,w
    #return NotImplementedError



def ridge_regression(y, tx, lambda_):
    "Ridge regression using normal equations"
    N,D = tx.shape
    A = tx.T.dot(tx) + 2 * N * lambda_ * np.identity(D)
    b = tx.T.dot(y)
    return np.linalg.solve(A, b)

    #return NotImplementedError



def sigmoid (t):
    return 1.0 / (1 + np.exp(-t))


def calculate_loss(y, tx, w):
    pred = sigmoid(tx.dot(w))
    loss = y.T.dot(np.log(pred)) + (1 - y).T.dot(np.log(1 - pred))
    return np.squeeze(-loss).item() * (1 / y.shape[0])

def calculate_gradient(y, tx, w):
    pred = sigmoid(tx.dot(w))
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])
    return grad



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    "Logistic regression using gradient descent y in {0,1}"
    threshold = 1e-8
    w = initial_w
    losses = []
    for i in range (max_iters):
        loss = calculate_loss(y, tx, w)
        grad = calculate_gradient(y, tx, w)
        w -= gamma * grad
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
    return losses[-1], w


    #return NotImplementedError



def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression 
    using gradient descent y in {0,1} w regularization term λ∥w∥^2"""
    w = initial_w
    losses = []
    thereshold = 1e-8
    for i in range(max_iters):
        loss = calculate_loss(y, tx, w) + lambda_ * np.sum(w ** 2)
        grad = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w -= gamma * grad
        losses.append(loss)
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break
        return losses[-1], w
    
    #return NotImplementedError