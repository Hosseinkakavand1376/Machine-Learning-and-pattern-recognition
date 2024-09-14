import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def vRow(v):
    return v.reshape((1, v.size))

def vcol(v):
    return v.reshape((v.size, 1))

# Define the Linear SVM functions
def compute_H(X, y, K):
    Z = y * 2.0 - 1.0  # Convert labels to +1/-1
    X_ext = np.vstack([X, np.ones((1, X.shape[1])) * K])
    H = np.dot(X_ext.T, X_ext) * vcol(Z) * vRow(Z)
    return H, X_ext, Z

def dual_objective_linear(alpha, H):
    Ha = H @ alpha
    loss = 0.5 * (alpha @ Ha) - alpha.sum()
    return loss

def dual_gradient_linear(alpha, H):
    grad = H @ alpha - np.ones(alpha.size)
    return grad
def primal_objective_linear(w_hat, X_ext, Z, C):
    return 0.5 * np.linalg.norm(w_hat) ** 2 + C * np.maximum(0, 1 - Z * np.dot(w_hat, X_ext)).sum()


def train_linear_svm(X, y, C, K):
    H, X_ext, Z = compute_H(X, y, K)
    bounds = [(0, C) for _ in range(X_ext.shape[1])]
    result = fmin_l_bfgs_b(dual_objective_linear, np.zeros(X_ext.shape[1]), fprime=dual_gradient_linear, args=(H,),
                           bounds=bounds, factr=1.0)
    alpha = result[0]

    w_hat = (vRow(alpha) * vRow(Z) * X_ext).sum(1)
    w, b = w_hat[0:X.shape[0]], w_hat[-1] * K

    primal_loss = primal_objective_linear(w_hat, X_ext, Z, C)
    dual_loss = -dual_objective_linear(alpha, H)
    duality_gap = primal_loss - dual_loss

    return w, b, primal_loss, dual_loss, duality_gap

def evaluate_svm(X, y, w, b):
    scores = np.dot(w.T, X) + b
    predictions = np.sign(scores)
    accuracy = np.mean(predictions == (y * 2.0 - 1.0))
    return accuracy

# Kernel Functions
def polynomial_kernel(degree=2, c=1):
    def polykernel(X1, X2):
        return (np.dot(X1.T, X2) + c) ** degree

    return polykernel

def rbf_kernel(gamma=1):
    def rbfkernel(X1, X2):
        dists = np.sum(X1 ** 2, axis=0, keepdims=True).T + np.sum(X2 ** 2, axis=0) - 2 * np.dot(X1.T, X2)
        return np.exp(-gamma * dists)

    return rbfkernel

# Dual SVM for Kernel
def compute_H_kernel(X, y, kernel_func, eps):
    Z = y * 2.0 - 1.0  # Convert labels to +1/-1
    H = kernel_func(X, X) + eps  # Add epsilon for regularization
    H = H * vcol(Z) * vRow(Z)
    return H


def dual_objective(alpha, H):
    Ha = H @ vcol(alpha)
    loss = 0.5 * (vRow(alpha) @ Ha).ravel() - alpha.sum()
    return loss.item()  # Extract scalar value


def dual_gradient(alpha, H):
    Ha = H @ vcol(alpha)
    grad = Ha.ravel() - np.ones(alpha.size)
    return grad


def primal_objective_kernel(alpha, H, C):
    Z = alpha * np.sum(H, axis=1)
    return 0.5 * np.dot(Z, alpha) + C * np.maximum(0, 1 - Z).sum()


def train_kernel_svm(X, y, C, kernel_func, eps):
    if C == 0:
        return np.zeros(X.shape[1])  # When C=0, the solution is trivial

    H = compute_H_kernel(X, y, kernel_func, eps)
    bounds = [(0, C) for _ in range(X.shape[1])]
    result = fmin_l_bfgs_b(dual_objective, np.zeros(X.shape[1]), fprime=dual_gradient, args=(H,), bounds=bounds,
                           factr=1.0)
    alpha = result[0]

    dual_loss_value = -(dual_objective(alpha, H))
    primal_loss_value = primal_objective_kernel(alpha, H, C)
    duality_gap = primal_loss_value - dual_loss_value

    return alpha, primal_loss_value, dual_loss_value, duality_gap


def evaluate_kernel_svm(X_train, y_train, X_test, alpha, kernel_func, eps):
    Z = y_train * 2.0 - 1.0
    support_indices = np.where(alpha > 1e-5)[0]
    support_vectors = X_train[:, support_indices]
    support_alphas = alpha[support_indices]
    support_labels = Z[support_indices]

    K_test = kernel_func(support_vectors, X_test) + eps
    scores = np.dot(support_alphas * support_labels, K_test)
    predictions = np.sign(scores)
    return predictions, scores
