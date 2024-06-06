import numpy as np
import matplotlib.pyplot as plt
import scipy


def vRow(v):
    return v.reshape((1, v.size))


def vcol(v):
    return v.reshape((v.size, 1))


def logpdf_GAU_ND(X, mu, C):
    XC = X - mu
    M = X.shape[0]
    const = -0.5 * M * np.log(2 * np.pi)
    log_det_C = np.linalg.slogdet(C)[1]
    invC = np.linalg.inv(C)
    exponent = -0.5 * np.sum(XC * np.dot(invC, XC), axis=0)
    return const - 0.5 * log_det_C + exponent


def split_db_2tol(D, L, seed=0):
    nTrain = int(D.shape[1] * (2.0 / 3.0))
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[:nTrain]
    idxTest = idx[nTrain:]

    DTR = D[:, idxTrain]
    DTS = D[:, idxTest]
    LTR = L[idxTrain]
    LTS = L[idxTest]
    return (DTR, LTR), (DTS, LTS)


def statistics_computations(D):
    """Dataset mean"""
    mu = vcol(D.mean(1))
    """Centered data"""
    centered_D = D - mu
    """Covariance matrix"""
    C = ((D - mu) @ (D - mu).T) / float(D.shape[1])
    return mu, C


def load_iris():
    import sklearn.datasets
    return sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']


def compute_MVG_ml_estimates(D, L):
    print("Computing MVG: ")
    unique_classes = np.unique(L)
    mvg_ml_estimates = {}
    for cls in unique_classes:
        D_cls = D[:, L == cls]
        mu_ml, sigma_ml = statistics_computations(D_cls)
        mvg_ml_estimates[cls] = (mu_ml, sigma_ml)
        print(f"Class {cls} mean vector:")
        print(mvg_ml_estimates[cls][0])
        print(f"Class {cls} covariance matrix:")
        print(mvg_ml_estimates[cls][1])
    return mvg_ml_estimates


def compute_naive_bayes_ml_estimates(D, L):
    print("Computing naive Bayes: ")
    unique_classes = np.unique(L)
    naive_bayes_ml_estimates = {}
    for cls in unique_classes:
        D_cls = D[:, L == cls]
        mu_cls, sigma_cls = statistics_computations(D_cls)
        naive_bayes_ml_estimates[cls] = (mu_cls, sigma_cls * np.eye(D.shape[0]))
        print(f"Class {cls} mean vector:")
        print(naive_bayes_ml_estimates[cls][0])
        print(f"Class {cls} covariance matrix:")
        print(naive_bayes_ml_estimates[cls][1])
    return naive_bayes_ml_estimates


def compute_tied_ml_estimates(D, L):
    print("Computing tied : ")
    unique_classes = np.unique(L)
    tied_ml_estimates = {}
    means = {}
    sigma_tied = 0
    for cls in unique_classes:
        D_cls = D[:, L == cls]
        mu_cls, sigma_cls = statistics_computations(D_cls)
        sigma_tied += D_cls.shape[1] * sigma_cls
        means[cls] = mu_cls
    sigma_tied /= D.shape[1]
    for cls in unique_classes:
        tied_ml_estimates[cls] = (means[cls], sigma_tied)
        print(f"Class {cls} mean vector:")
        print(tied_ml_estimates[cls][0])
        print(f"Class {cls} covariance matrix:")
        print(tied_ml_estimates[cls][1])
    return tied_ml_estimates


def compute_error_rate(log_probs, LTS):
    """Compute error rate given log probabilities and true labels."""
    predicted_labels = np.argmax(log_probs, axis=0)
    correct_predictions = np.sum(predicted_labels == LTS)
    error_rate = 1 - (correct_predictions / LTS.size)
    return error_rate


if __name__ == "__main__":
    D, L = load_iris()
    (DTR, LTR), (DTS, LTS) = split_db_2tol(D, L)

    # Compute ML estimates
    mvg_ml_estimates = compute_MVG_ml_estimates(DTR, LTR)
    naive_bayes_estimates = compute_naive_bayes_ml_estimates(DTR, LTR)
    tied_ml_estimates = compute_tied_ml_estimates(DTR, LTR)

    # Initialize log-probability matrices
    log_mvg = np.zeros((len(np.unique(LTR)), DTS.shape[1]))
    log_naive_bayes = np.zeros((len(np.unique(LTR)), DTS.shape[1]))
    log_tied = np.zeros((len(np.unique(LTR)), DTS.shape[1]))

    # Compute log-probabilities
    for i in np.unique(LTR):
        log_mvg[i] = logpdf_GAU_ND(DTS, mvg_ml_estimates[i][0], mvg_ml_estimates[i][1])
        log_naive_bayes[i] = logpdf_GAU_ND(DTS, naive_bayes_estimates[i][0], naive_bayes_estimates[i][1])
        log_tied[i] = logpdf_GAU_ND(DTS, tied_ml_estimates[i][0], tied_ml_estimates[i][1])

    # Compute error rates
    error_rate_mvg = compute_error_rate(log_mvg, LTS)
    error_rate_naive_bayes = compute_error_rate(log_naive_bayes, LTS)
    error_rate_tied = compute_error_rate(log_tied, LTS)

    print(f"MVG Error Rate: {error_rate_mvg*100:.2f}%")
    print(f"Naive Bayes Error Rate: {error_rate_naive_bayes*100:.2f}%")
    print(f"Tied Covariance Error Rate: {error_rate_tied*100:.2f}%")
