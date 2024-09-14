import numpy as np
import scipy

# Define helper functions
def load_Data(file_name):
    FeaturesLists = []
    Labels = []
    with open(file_name, 'r') as f:
        for line in f:
            features = line.split(',')[0:-1]
            features = np.array([float(i) for i in features]).reshape(-1, 1)
            FeaturesLists.append(features)
            label = int(line.split(',')[-1])
            Labels.append(label)
    return np.hstack(FeaturesLists), np.array(Labels)

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
    mu = np.mean(D, axis=1).reshape(-1, 1)
    centered_D = D - mu
    C = (centered_D @ centered_D.T) / float(D.shape[1])
    var = np.var(D, axis=1).reshape(-1, 1)
    std = np.std(D, axis=1).reshape(-1, 1)
    return mu, centered_D, C, var, std

def pca(D, nfeatures):
    mu, centered_D, C, var, std = statistics_computations(D)
    eigenValues, eigenVectors = scipy.linalg.eigh(C)
    sorted_eigenValues = np.argsort(eigenValues)[::-1]
    sorted_eigenVectors = eigenVectors[:, sorted_eigenValues]
    selected_eigenVectors = sorted_eigenVectors[:, :nfeatures]
    return selected_eigenVectors

def lda(D, L, n_features):
    unique_labels = np.unique(L)
    mu, centered_D, C, var, std = statistics_computations(D)
    SW = np.zeros((D.shape[0], D.shape[0]))
    SB = np.zeros((D.shape[0], D.shape[0]))
    for label in unique_labels:
        D_i = D[:, L == label]
        mean_i = np.mean(D_i, axis=1, keepdims=True)
        SW += (D_i - mean_i).dot((D_i - mean_i).T) / (D.shape[1])
        n_i = D_i.shape[1]
        mean_diff = mean_i - mu
        SB += n_i * mean_diff.dot(mean_diff.T) / D.shape[1]
    _, eigenVectors = scipy.linalg.eigh(SB, SW)
    selected_eigenVectors = eigenVectors[:, ::-1][:, :n_features]
    return selected_eigenVectors

def evaluate_threshold(threshold, DVAL_lda, LVAL):
    PVAL = np.zeros(LVAL.shape, dtype=np.int32)
    PVAL[DVAL_lda[0] >= threshold] = 1
    PVAL[DVAL_lda[0] < threshold] = 0
    error_rate = np.mean(PVAL != LVAL)
    accuracy = np.sum(PVAL == LVAL)
    return error_rate, accuracy

def confusion_matrix(y_pred, y_true):
    K = y_true.max() + 1
    conf_matrix = np.zeros((K, K), dtype=np.int32)
    for i in range(y_true.size):
        conf_matrix[y_pred[i], y_true[i]] += 1
    return conf_matrix

def actdcf(llr, y_true, pi, cost_fn, cost_fp, Normalize=True):
    conf_matrix = confusion_matrix(llr, y_true)
    FN = conf_matrix[0, 1]
    FP = conf_matrix[1, 0]
    TP = conf_matrix[0, 0]
    TN = conf_matrix[1, 1]
    P_fn = FN / (FN + TN) if (FN + TN) != 0 else 0
    P_fp = FP / (FP + TP) if (FP + TP) != 0 else 0
    DCF = pi * cost_fn * P_fn + (1 - pi) * cost_fp * P_fp
    if Normalize:
        return DCF / min(pi * cost_fn, (1 - pi) * cost_fp)
    else:
        return DCF

def compute_minDCF(llr, y_true, pi, cost_fn, cost_fp, returnThreshold=False):
    y_scores = llr
    thresholds = np.concatenate([np.array([-np.inf]), y_scores, np.array([np.inf])])
    th = None
    min_DCF = None
    for threshold in thresholds:
        y_pred = np.int32(y_scores > threshold)
        dcf = actdcf(y_pred, y_true, pi, cost_fn, cost_fp, Normalize=True)
        if min_DCF is None or dcf < min_DCF:
            min_DCF = dcf
            th = threshold
    if returnThreshold:
        return min_DCF, th
    else:
        return min_DCF

def compute_empirical_bayes_decisions_binary(llr, pi1, Cfn, Cfp):
    threshold = -np.log((pi1 * Cfn) / ((1 - pi1) * Cfp))
    predictions = np.int32(llr > threshold)
    return predictions

def compute_DCF(llr, y_true, prior, Cfn, Cfp, normalize=True):
    predicted = compute_empirical_bayes_decisions_binary(llr, prior, Cfn, Cfp)
    return actdcf(predicted, y_true, prior, Cfn, Cfp, normalize)
