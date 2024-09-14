import numpy as np
from utils import *
def vRow(v):
    return v.reshape((1, v.size))

def vcol(v):
    return v.reshape((v.size, 1))

def logpdf_GAU_ND(X, mu, C):
    if C.ndim == 0:
        return -np.inf * np.ones(X.shape[1])
    XC = X - mu.reshape(-1, 1)
    M = X.shape[0]
    const = -0.5 * M * np.log(2 * np.pi)
    log_det_C = np.linalg.slogdet(C)[1]
    invC = np.linalg.inv(C)
    exponent = -0.5 * np.sum(XC * np.dot(invC, XC), axis=0)
    return const - 0.5 * log_det_C + exponent
def compute_ml_estimates(D, L):
    unique_classes = np.unique(L)
    ml_estimates = {}
    for cls in unique_classes:
        D_cls = D[:, L == cls]
        mu_ml = np.mean(D_cls, axis=1)
        sigma_ml = np.var(D_cls, axis=1)
        ml_estimates[cls] = (mu_ml, sigma_ml)
    return ml_estimates

def compute_empirical_parameters(D, L):
    unique_classes = np.unique(L)
    parameters = {}
    for cls in unique_classes:
        D_cls = D[:, L == cls]
        mu_cls = np.mean(D_cls, axis=1, keepdims=True)
        sigma_cls = np.cov(D_cls)
        parameters[cls] = (mu_cls, sigma_cls)
    return parameters

def compute_tied_covariance(D, L, parameters):
    sigma_tied = np.zeros(parameters[0][1].shape)
    for cls, (mu_cls, sigma_cls) in parameters.items():
        D_cls = D[:, L == cls]
        sigma_tied += D_cls.shape[1] * sigma_cls
    sigma_tied /= D.shape[1]
    return sigma_tied

def compute_naive_bayes_parameters(D, L):
    unique_classes = set(L)
    parameters = {}
    for cls in unique_classes:
        D_cls = D[:, L == cls]
        mu, _, C, _, _ = statistics_computations(D_cls)
        parameters[cls] = (mu, C * np.eye(D.shape[0]))
    return parameters

def MVG_classifier(DTS, parameters, priors):
    S = np.zeros((len(parameters), DTS.shape[1]))
    for cls, (mu_cls, sigma_cls) in parameters.items():
        S[cls, :] = logpdf_GAU_ND(DTS, mu_cls, sigma_cls) + np.log(priors[cls] + 1e-10)
    return S

def compute_accuracy(S, LTS):
    predicted_labels = np.argmax(S, axis=0)
    accuracy = np.sum(predicted_labels == LTS) / LTS.shape[0]
    return accuracy

def evaluate_models(DTR, LTR, DTS, LTS, priors, costs):
    results = {}
    llr_values = {}

    # MVG Classifier
    parameters = compute_empirical_parameters(DTR, LTR)
    S = MVG_classifier(DTS, parameters, priors)
    accuracy = compute_accuracy(S, LTS)
    llr = logpdf_GAU_ND(DTS, parameters[1][0], parameters[1][1]) - logpdf_GAU_ND(DTS, parameters[0][0], parameters[0][1])
    min_dcf = compute_minDCF(llr, LTS, priors[1], costs[0], costs[1])
    dcf = compute_DCF(llr, LTS, priors[1], costs[0], costs[1])
    unnormal_dcf = compute_DCF(llr, LTS, priors[1], costs[0], costs[1], normalize=False)
    results['MVG'] = {'accuracy': accuracy, 'DCF': dcf, 'min_dcf': min_dcf}
    llr_values['MVG'] = llr
    print(f'MVG Classifier Accuracy: {accuracy * 100:.2f}%, Normalized DCF: {dcf:.4f}, Unnormalized DCF: {unnormal_dcf:.4f}, Min DCF: {min_dcf:.4f}')

    # Tied Covariance Gaussian Classifier
    sigma_tied = compute_tied_covariance(DTR, LTR, parameters)
    for cls in parameters:
        parameters[cls] = (parameters[cls][0], sigma_tied)
    S = MVG_classifier(DTS, parameters, priors)
    accuracy = compute_accuracy(S, LTS)
    llr = logpdf_GAU_ND(DTS, parameters[1][0], sigma_tied) - logpdf_GAU_ND(DTS, parameters[0][0], sigma_tied)
    min_dcf = compute_minDCF(llr, LTS, priors[1], costs[0], costs[1])
    dcf = compute_DCF(llr, LTS, priors[1], costs[0], costs[1])
    unnormal_dcf = compute_DCF(llr, LTS, priors[1], costs[0], costs[1], normalize=False)
    results['Tied MVG'] = {'accuracy': accuracy, 'DCF': dcf, 'min_dcf': min_dcf}
    llr_values['Tied MVG'] = llr
    print(f'Tied Covariance Gaussian Classifier Accuracy: {accuracy * 100:.2f}%, Normalized DCF: {dcf:.4f}, Unnormalized DCF: {unnormal_dcf:.4f}, Min DCF: {min_dcf:.4f}')

    # Naive Bayes Gaussian Classifier
    parameters = compute_naive_bayes_parameters(DTR, LTR)
    S = np.zeros((len(parameters), DTS.shape[1]))
    accuracy = compute_accuracy(S, LTS)
    llr = logpdf_GAU_ND(DTS, parameters[1][0], parameters[1][1]) - logpdf_GAU_ND(DTS, parameters[0][0], parameters[0][1])
    min_dcf = compute_minDCF(llr, LTS, priors[1], costs[0], costs[1])
    dcf = compute_DCF(llr, LTS, priors[1], costs[0], costs[1])
    unnormal_dcf = compute_DCF(llr, LTS, priors[1], costs[0], costs[1], normalize=False)
    results['Naive Bayes'] = {'accuracy': accuracy, 'DCF': dcf, 'min_dcf': min_dcf}
    llr_values['Naive Bayes'] = llr
    print(f'Naive Bayes Gaussian Classifier Accuracy: {accuracy * 100:.2f}%, Normalized DCF: {dcf:.4f}, Unnormalized DCF: {unnormal_dcf:.4f}, Min DCF: {min_dcf:.4f}')
    return results, llr_values
