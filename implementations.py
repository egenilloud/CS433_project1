"""Regression methods for CS433 Project 1."""

import numpy as np

def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    Args:
        y: shape=(N, ), target values.
        tx: shape=(N, D), input features.
        w: shape=(D, ), model weights.

    Returns:
        grad: shape=(D, ), gradient of the MSE loss.
        err : shape=(N, ), residuals.
    """

    err = y - tx @ w
    grad = -(tx.T @ err) / len(err)
    return grad, err

def calculate_mse(e):
    """Compute MSE.
    
    Args:
        err: shape=(N,), residuals.

    Returns:
        loss: 0.5 * mean(err**2).
    """
    return 1 / 2 * np.mean(e**2)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent
    
    Args : 
        y : shape (N,) : target values
        tx : shape (N,D) : input data
        initial_w : shape (D,) : initial weights
        max_iters : int, number of iterations
        gamma : float, learning rate

    Returns:
        w: shape (D,), final model weights.
        loss: 0-dim array, final MSE on all observations.
    
    """
    w = initial_w

    for _ in range(max_iters):
        # compute loss, gradient
        grad, _ = compute_gradient(y, tx, w)    
        # update w by gradient descent
        w = w - gamma * grad
    
    loss = calculate_mse(y - tx @ w)
    return w, np.asarray(loss)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """
    Fit linear regression using stochastic gradient descent.

    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        initial_w: shape=(D, )
        max_iters: number of iterations
        gamma: a scalar denoting the learning rate

    Returns:
        w: final weight, shape=(D,)
        loss: final MSE loss, scalar
    """
    w = initial_w
    n = len(y)

    for _ in range(max_iters):
        # Randomly select one observation
        i = np.random.randint(n)

        # Compute a stochastic gradient and loss
        grad, _ = compute_gradient(y[i:i+1], tx[i:i+1], w)
        # Update w 
        w = w - gamma * grad
        
    # Calculate loss
    loss = calculate_mse(y - tx @ w)
    return w, np.asarray(loss)
        


def least_squares(y, tx):
    """Fit linear regression using the least-squares normal equations.
    
    Args:
        y: shape (N,), target values.
        tx: shape (N, D), input features.
    
    Returns:
        w: shape (D,), fitted model weights.
        loss: 0-dimensional array.
    """
    A = tx.T @ tx
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    e = y - tx @ w
    loss = calculate_mse(e)

    return w, np.asarray(loss)



def ridge_regression(y, tx, lambda_):
    """Fit L2-regularized linear regression using normal equations.
    
        Args:
            y: shape (N,), target values.
            tx: shape (N, D), input features.
            lambda_: float, nonnegative L2 regularization strength.
    
        Returns:
            w: shape (D,), fitted model weights.
            loss: 0-dimensional array, MSE without the L2 penalty.
        """
    N, D = tx.shape
    A = tx.T @ tx + 2 * N * lambda_ * np.identity(D)
    b = tx.T @ y
    w = np.linalg.solve(A, b)
    e = y - tx @ w
    loss = calculate_mse(e)
    return w, np.asarray(loss)


def sigmoid (t):
    """Apply the logistic sigmoid function elementwise."""
    return 1.0 / (1 + np.exp(-t))


def calculate_loss_logistic(y, tx, w):
    pred = sigmoid(tx @ w)
    loss = y.T @ np.log(pred) + (1 - y).T @ np.log(1 - pred)
    return np.squeeze(-loss).item() * (1 / y.shape[0])

def calculate_gradient_logistic(y, tx, w):
    pred = sigmoid(tx @ w)
    grad = tx.T.dot(pred - y) * (1 / y.shape[0])
    return grad



def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent y in {0,1}
        Stop early when the absolute change in loss falls below 1e-8.
    
        Args:
            y: shape (N,), binary labels in {0, 1}.
            tx: shape (N, D), input features.
            initial_w: shape (D,), initial model weights.
            max_iters: int, maximum number of gradient updates.
            gamma: float, learning rate.
    
        Returns:
            w: shape (D,), final model weights.
            loss: 0-dimensional array, final mean logistic loss.
        """
    threshold = 1e-8
    w = initial_w
    previous_loss = None

    for i in range (max_iters):
        grad = calculate_gradient_logistic(y, tx, w)
        w -= gamma * grad
        loss = calculate_loss_logistic(y, tx, w)
        
        if previous_loss is not None and np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss

    # Calculate loss in case of max_iters 0    
    loss = np.asarray(calculate_loss_logistic(y, tx, w))
    
    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression 
    using gradient descent y in {0,1} w regularization term λ∥w∥^2
    
    Args:
        y: shape (N,), binary labels in {0, 1}.
        tx: shape (N, D), input features.
        lambda_: float, L2 regularization strength.
        initial_w: shape (D,), initial model weights.
        max_iters: int, maximum number of gradient updates.
        gamma: float, learning rate.
    
        Returns:
        w: shape (D,), final model weights.
        loss: 0-dimensional array, final mean logistic loss without L2.
        """
    w = initial_w
    threshold = 1e-8
    previous_loss = None

    for i in range(max_iters):
        grad = calculate_gradient_logistic(y, tx, w) + 2 * lambda_ * w
        w -= gamma * grad
        loss = calculate_loss_logistic(y, tx, w) + lambda_ * np.sum(w ** 2)

        if previous_loss is not None and np.abs(loss - previous_loss) < threshold:
            break
        previous_loss = loss

    # Calculate loss in case of max_iters 0  
    loss = np.asarray(calculate_loss_logistic(y, tx, w))
    return w, loss
    
