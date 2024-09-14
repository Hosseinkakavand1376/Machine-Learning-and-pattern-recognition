import os
from SVM import *
from MVG import *
from logistic import *
from plots import *
from GMM import *

def extract_train_val_folds_from_ary(X, idx):
    return np.hstack([X[jdx::KFOLD] for jdx in range(KFOLD) if jdx != idx]), X[idx::KFOLD]


def calibration(DTR, LTR, DVAL, LVAL, D_eval, pT):
    print(f'pT:', pT)
    best_mindcf = float('inf')
    best_model = None

    # Load GMM model parameters
    gmm_model = np.load('pi1_0.1_Cfn_1.0_Cfp_1.0/best_model_1_Best GMM Model.npy', allow_pickle=True).item()
    gmm0 = gmm_model['gmm0']
    gmm1 = gmm_model['gmm1']
    llr_val_gmm = logpdf_GMM(DVAL, gmm1) - logpdf_GMM(DVAL, gmm0)
    llr_eval_gmm = logpdf_GMM(D_eval, gmm1) - logpdf_GMM(D_eval, gmm0)

    # Load SVM model parameters
    svm_model = np.load('pi1_0.1_Cfn_1.0_Cfp_1.0/best_model_2_Best SVM Model.npy', allow_pickle=True).item()
    alpha = svm_model['alpha']
    gamma = svm_model['gamma']
    kernel_func = rbf_kernel(gamma)
    _, llr_val_svm = evaluate_kernel_svm(DTR, LTR, DVAL, alpha, kernel_func, 1.0)
    _, llr_eval_svm = evaluate_kernel_svm(DTR, LTR, D_eval, alpha, kernel_func, 1.0)

    # Load LR model parameters
    lr_model = np.load('pi1_0.1_Cfn_1.0_Cfp_1.0/best_model_3_Best Regression Model.npy', allow_pickle=True).item()
    w = lr_model['w']
    b = lr_model['b']
    prior = (LTR == 1).sum() / LTR.size
    llr_val_lr = np.dot(w.T, expand_features(DVAL)) + b - np.log(prior / (1 - prior))
    llr_eval_lr = np.dot(w.T, expand_features(D_eval)) + b - np.log(prior / (1 - prior))

    # Calibrate GMM
    print('Calibrating GMM')
    calibrated_scores_gmm = []
    labels_gmm = []
    labels = LVAL
    for foldIdx in range(KFOLD):
        SCAL, SVAL = extract_train_val_folds_from_ary(llr_val_gmm, foldIdx)
        LCAL, LVAL_fold = extract_train_val_folds_from_ary(labels, foldIdx)
        w, b = trainWeightedLogRegBinary(vRow(SCAL), LCAL, 0, pT)
        calibrated_SVAL = (w.T @ vRow(SVAL) + b - np.log(pT / (1 - pT))).ravel()
        calibrated_scores_gmm.append(calibrated_SVAL)
        labels_gmm.append(LVAL_fold)
    calibrated_scores_gmm = np.hstack(calibrated_scores_gmm)
    labels_gmm = np.hstack(labels_gmm)
    mindcf = compute_minDCF(calibrated_scores_gmm, labels_gmm, 0.1, 1.0, 1.0)
    DCF_validation = compute_DCF(calibrated_scores_gmm, labels_gmm, 0.1, 1.0, 1.0)
    print('Val DCF no cal:',compute_DCF(llr_val_gmm, LVAL, 0.1, 1.0, 1.0))
    print('Val minDCF no cal: ', compute_minDCF(llr_val_gmm, LVAL, 0.1, 1.0, 1.0))
    print('Calibrated Validation DCF: ', DCF_validation)
    print(f'Calibrated Validation minDCF: {mindcf}')
    # plot_bayes_error(calibrated_scores_gmm, labels_gmm, 1.0, 1.0, title=f'Val calibration for GMM - pT: {pT}')

    w, b = trainWeightedLogRegBinary(vRow(llr_val_gmm), labels, 0, pT)
    calibrated_eval_scores_gmm = (w.T @ vRow(llr_eval_gmm) + b - np.log(pT / (1 - pT))).ravel()
    gmm_mindcf = compute_minDCF(calibrated_eval_scores_gmm, L_eval, 0.1, 1.0, 1.0)
    print(f'Evaluation minDCF cal: {gmm_mindcf}')
    print(f'Evaluation DCF cal: {compute_DCF(calibrated_eval_scores_gmm, L_eval, 0.1, 1.0, 1.0)}')
    if gmm_mindcf < best_mindcf:
        best_mindcf = gmm_mindcf
        best_model = "GMM"
    plot_bayes_error(llr_eval_gmm, L_eval, 1.0, 1.0, title=f'Eval no calibration for GMM - {pT}')
    plot_bayes_error(calibrated_eval_scores_gmm, L_eval, 1.0, 1.0, title=f'Eval calibration for GMM - {pT}')

    # Calibrate SVM
    print('Calibrating SVM')
    calibrated_scores_svm = []
    labels_svm = []
    for foldIdx in range(KFOLD):
        SCAL, SVAL = extract_train_val_folds_from_ary(llr_val_svm, foldIdx)
        LCAL, LVAL_fold = extract_train_val_folds_from_ary(labels, foldIdx)
        w, b = trainWeightedLogRegBinary(vRow(SCAL), LCAL, 0, pT)
        calibrated_SVAL = (w.T @ vRow(SVAL) + b - np.log(pT / (1 - pT))).ravel()
        calibrated_scores_svm.append(calibrated_SVAL)
        labels_svm.append(LVAL_fold)
    calibrated_scores_svm = np.hstack(calibrated_scores_svm)
    labels_svm = np.hstack(labels_svm)
    mindcf = compute_minDCF(calibrated_scores_svm, labels_svm, 0.1, 1.0, 1.0)
    DCF_validation = compute_DCF(calibrated_scores_svm, labels_svm, 0.1, 1.0, 1.0)
    print('Val DCF no cal:', compute_DCF(llr_val_svm, LVAL, 0.1, 1.0, 1.0))
    print('Val minDCF no cal: ', compute_minDCF(llr_val_svm, LVAL, 0.1, 1.0, 1.0))
    print('Calibrated Validation DCF: ', DCF_validation)
    print(f'Calibrated Validation minDCF: {mindcf}')
    # plot_bayes_error(calibrated_scores_svm,labels_svm, 1.0, 1.0, title=f'Val calibration for SVM - pT: {pT}')

    w, b = trainWeightedLogRegBinary(vRow(llr_val_svm), labels, 0, pT)
    calibrated_eval_scores_svm = (w.T @ vRow(llr_eval_svm) + b - np.log(pT / (1 - pT))).ravel()
    SVM_mindcf = compute_minDCF(calibrated_eval_scores_svm, L_eval, 0.1, 1.0, 1.0)
    print(f'Evaluation minDCF cal: {SVM_mindcf}')
    print(f'Evaluation DCF cal: {compute_DCF(calibrated_eval_scores_svm, L_eval, 0.1, 1.0, 1.0)}')
    if SVM_mindcf < best_mindcf:
        best_mindcf = SVM_mindcf
        best_model = "SVM"
    plot_bayes_error(llr_eval_svm, L_eval, 1.0, 1.0, title=f'Eval no calibration for SVM - {pT}')
    plot_bayes_error(calibrated_eval_scores_svm, L_eval, 1.0, 1.0, title=f'Eval calibration for SVM - {pT}')

    # Calibrate LR
    print('Calibrating LR')
    calibrated_scores_lr = []
    labels_lr = []
    for foldIdx in range(KFOLD):
        SCAL, SVAL = extract_train_val_folds_from_ary(llr_val_lr, foldIdx)
        LCAL, LVAL_fold = extract_train_val_folds_from_ary(labels, foldIdx)
        w, b = trainWeightedLogRegBinary(vRow(SCAL), LCAL, 0, pT)
        calibrated_SVAL = (w.T @ vRow(SVAL) + b - np.log(pT / (1 - pT))).ravel()
        calibrated_scores_lr.append(calibrated_SVAL)
        labels_lr.append(LVAL_fold)
    calibrated_scores_lr = np.hstack(calibrated_scores_lr)
    labels_lr = np.hstack(labels_lr)
    mindcf = compute_minDCF(calibrated_scores_lr, labels_lr, 0.1, 1.0, 1.0)
    DCF_validation = compute_DCF(calibrated_scores_lr, labels_lr, 0.1, 1.0, 1.0)
    print('Val DCF no cal:', compute_DCF(llr_val_lr, LVAL, 0.1, 1.0, 1.0))
    print('Val minDCF no cal: ', compute_minDCF(llr_val_lr, LVAL, 0.1, 1.0, 1.0))
    print('Calibrated Validation DCF: ', DCF_validation)
    print(f'Calibrated Validation minDCF: {mindcf}')
    # plot_bayes_error(calibrated_scores_lr, labels_lr, 1.0, 1.0, title=f'Val calibration for LR - pT: {pT}')
    w, b = trainWeightedLogRegBinary(vRow(llr_val_lr), labels, 0, pT)
    calibrated_eval_scores_lr = (w.T @ vRow(llr_eval_lr) + b - np.log(pT / (1 - pT))).ravel()
    LR_mindcf = compute_minDCF(calibrated_eval_scores_lr, L_eval, 0.1, 1.0, 1.0)
    print(f'Evaluation minDCF cal: {LR_mindcf}')
    print(f'Evaluation DCF cal: {compute_DCF(calibrated_eval_scores_lr, L_eval, 0.1, 1.0, 1.0)}')
    if LR_mindcf < best_mindcf:
        best_mindcf = LR_mindcf
        best_model = "LR"
    plot_bayes_error(llr_eval_lr, L_eval, 1.0, 1.0, title=f'Eval no calibration for LR - {pT}')
    plot_bayes_error(calibrated_eval_scores_lr, L_eval, 1.0, 1.0, title=f'Eval calibration for LR - {pT}')

    # Fusion Model
    print(f'Fusion model with pT:', pT)
    fusedScores = []
    fusedLabels = []
    for foldIdx in range(KFOLD):
        SCAL1, SVAL1 = extract_train_val_folds_from_ary(llr_val_lr, foldIdx)
        SCAL2, SVAL2 = extract_train_val_folds_from_ary(llr_val_svm, foldIdx)
        SCAL3, SVAL3 = extract_train_val_folds_from_ary(llr_val_gmm, foldIdx)
        LCAL, LVAL_fold = extract_train_val_folds_from_ary(labels, foldIdx)
        SCAL = np.vstack([SCAL1, SCAL2, SCAL3])
        w, b = trainWeightedLogRegBinary(SCAL, LCAL, 0, pT)
        SVAL = np.vstack([SVAL1, SVAL2, SVAL3])
        calibrated_SVAL = (w.T @ SVAL + b - np.log(pT / (1 - pT))).ravel()
        fusedScores.append(calibrated_SVAL)
        fusedLabels.append(LVAL_fold)
    fusedScores = np.hstack(fusedScores)
    fusedLabels = np.hstack(fusedLabels)
    mindcf = compute_minDCF(fusedScores, fusedLabels, 0.1, 1.0, 1.0)
    DCF_validation = compute_DCF(fusedScores, fusedLabels, 0.1, 1.0, 1.0)
    print(f'Fusion model - Calibrated Validation minDCF: {mindcf}')
    print('Fusion model - Calibrated Validation DCF: ', DCF_validation)
    w, b = trainWeightedLogRegBinary(np.vstack([llr_val_lr, llr_val_svm, llr_val_gmm]),
                                     labels, 0, pT)
    calibrated_eval_scores_Fusion = (
                w.T @ np.vstack([llr_eval_lr, llr_eval_svm, llr_eval_gmm]) + b - np.log(pT / (1 - pT))).ravel()
    prediction = np.int32(calibrated_eval_scores_Fusion>0)
    error_rate = np.mean(prediction != L_eval)
    eval_minDCF = compute_minDCF(calibrated_eval_scores_Fusion, L_eval, 0.1, 1.0, 1.0)
    eval_DCF = compute_DCF(calibrated_eval_scores_Fusion, L_eval, 0.1, 1.0, 1.0)
    print(f'Evaluation minDCF cal: {eval_minDCF}')
    print(f'Evaluation DCF cal: {eval_DCF}')
    if eval_minDCF < best_mindcf:
        best_mindcf = eval_minDCF
        best_model = "Fusion"
    plot_bayes_error(calibrated_eval_scores_Fusion, L_eval, 1.0, 1.0, title=f'Eval calibration for Fusion - {pT}')

    return best_model, best_mindcf


D, L = load_Data('trainData.txt')
(DTR, LTR), (DVAL, LVAL) = split_db_2tol(D, L)
D_eval, L_eval = load_Data('evalData.txt')
D_eval_expand = expand_features(D_eval)
KFOLD = 5

# Calibrate and evaluate for each model and priors
for pT in [0.1, 0.5, 0.9]:
    model, DCF = calibration(DTR, LTR, DVAL, LVAL, D_eval, pT)
    print(f'Best model for pT={pT}: {model}, with DCF: {DCF}')
    print('\n')

