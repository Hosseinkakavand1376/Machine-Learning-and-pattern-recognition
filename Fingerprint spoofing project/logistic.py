import numpy as np
from scipy.optimize import fmin_l_bfgs_b

def vRow(v):
    return v.reshape((1, v.size))

def vcol(v):
    return v.reshape((v.size, 1))

def trainLogReg(data, label, l):
    Z = label * 2.0 - 1.0

    def logreg_obj(v):
        w, b = v[:-1], v[-1]
        s = np.dot(vcol(w).T, data).ravel() + b

        loss = (np.logaddexp(0, -Z * s)).mean()
        regularization = (l / 2 * np.linalg.norm(w) ** 2)
        J = regularization + loss
        gradient_w = (vRow(-Z / (1.0 + np.exp(Z * s))) * data).mean(1) + l * w.ravel()
        gradient_b = (-Z / (1.0 + np.exp(Z * s))).mean()
        G = np.hstack([gradient_w, np.array(gradient_b)])
        return J, G

    result = fmin_l_bfgs_b(logreg_obj, x0=np.zeros(data.shape[0] + 1))[0]
    return result[:-1], result[-1]

def trainWeightedLogRegBinary(data, label, l, pT):
    Z = label * 2.0 - 1.0  # Convert labels to -1 and 1

    w_True = pT / (Z > 0).sum()  # Weight for positive class
    w_False = (1 - pT) / (Z < 0).sum()  # Weight for negative class

    def logreg_obj(v):
        w, b = v[:-1], v[-1]
        s = np.dot(vcol(w).T, data).ravel() + b

        loss = np.logaddexp(0, -Z * s)
        loss[Z > 0] *= w_True  # Apply weights to loss
        loss[Z < 0] *= w_False

        gradient = -Z / (1.0 + np.exp(Z * s))
        gradient[Z > 0] *= w_True  # Apply weights to gradient
        gradient[Z < 0] *= w_False

        regularization = (l / 2 * np.linalg.norm(w) ** 2)
        J = regularization + loss.sum()

        gradient_w = (vRow(gradient) * data).sum(1) + l * w.ravel()
        gradient_b = gradient.sum()
        G = np.hstack([gradient_w, np.array(gradient_b)])
        return J, G

    result = fmin_l_bfgs_b(logreg_obj, x0=np.zeros(data.shape[0] + 1), approx_grad=False)
    return result[0][:-1], result[0][-1]

def expand_features(D):
    n_features, n_samples = D.shape
    D_expanded = np.zeros((n_features * (n_features + 1) // 2 + n_features, n_samples))
    idx = n_features
    for i in range(n_features):
        D_expanded[i, :] = D[i, :]
        for j in range(i, n_features):
            D_expanded[idx, :] = D[i, :] * D[j, :]
            idx += 1
    return D_expanded
