from implementations import *



def calculate_hessian(y, tx, w):
    """Compute the Hessian matrix of the logistic loss function.
    
    Args:
        y: shape (N,), binary labels in {0, 1}.
        tx: shape (N, D), input features.
        w: shape (D,), model weights.
    
    Returns:
        H: shape (D, D), Hessian matrix.
    """
    N = len(y)
    pred = sigmoid(tx @ w)
    S = np.diag(pred * (1 - pred))
    H = tx.T @ S @ tx / N
    return H


def Newton_one_step(y, tx, w, gamma):
    loss = calculate_loss_logistic(y, tx, w)
    gradient = calculate_gradient_logistic(y, tx, w)
    hessian = calculate_hessian(y, tx, w)
    w -= gamma * np.linalg.solve(hessian, gradient)

    return w, loss

def logistic_regression_newton(y, tx, initial_w, max_iters, gamma):
    """Fit logistic regression using Newton's method.
    
    Args:
        y: shape (N,), binary labels in {0, 1}.
        tx: shape (N, D), input features.
        initial_w: shape (D,), initial model weights.
        max_iters: int, maximum number of iterations.
        gamma: float, step size.
    
    Returns:
        w: shape (D,), fitted model weights.
        loss: 0-dimensional array, final loss value.
    """
    w = initial_w
    for n_iter in range(max_iters):
        w, loss = Newton_one_step(y, tx, w, gamma)
        
    return w, np.asarray(loss)