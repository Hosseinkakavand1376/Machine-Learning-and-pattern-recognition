import numpy as np
import scipy

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

def compute_mu_C(D):
    mu = vcol(D.mean(1))
    C = np.dot(D - mu, (D - mu).T) / D.shape[1]
    return mu, C

def logpdf_GMM(X, gmm):
    S = np.vstack([logpdf_GAU_ND(X, mu, C) + np.log(w) for w, mu, C in gmm])
    return scipy.special.logsumexp(S, axis=0)

def smooth_covariance_matrix(C, psi):
    U, s, Vh = np.linalg.svd(C)
    s[s < psi] = psi
    return U @ np.diag(s) @ U.T

def train_GMM_EM_Iteration(X, gmm, covType, psiEig=None):
    S = []
    for w, mu, C in gmm:
        logpdf_conditional = logpdf_GAU_ND(X, mu, C)
        logpdf_joint = logpdf_conditional + np.log(w)
        S.append(logpdf_joint)
    S = np.vstack(S)
    logdens = scipy.special.logsumexp(S, axis=0)
    gammaAllComponents = np.exp(S - logdens)

    gmmUpd = []
    for gIdx in range(len(gmm)):
        gamma = gammaAllComponents[gIdx]
        Z = gamma.sum()
        F = vcol((vRow(gamma) * X).sum(1))
        S = np.dot(vRow(gamma) * X, X.T)
        muUpd = F / Z
        CUpd = S / Z - np.dot(muUpd, muUpd.T)
        wUpd = Z / X.shape[1]
        if covType == 'diagonal':
            CUpd = np.diag(np.diag(CUpd))
        gmmUpd.append((wUpd, muUpd, CUpd))

    if covType == 'tied':
        CTied = sum(w * C for w, mu, C in gmmUpd) / len(gmmUpd)
        gmmUpd = [(w, mu, CTied) for w, mu, C in gmmUpd]

    if psiEig is not None:
        gmmUpd = [(w, mu, smooth_covariance_matrix(C, psiEig)) for w, mu, C in gmmUpd]

    return gmmUpd

def train_GMM_EM(X, gmm, covType, psiEig=None, epsLLAverage=1e-6, verbose=True):
    llOld = logpdf_GMM(X, gmm).mean()
    llDelta = None
    if verbose:
        print('GMM - it %3d - average ll %.8e' % (0, llOld))
    it = 1
    while (llDelta is None or llDelta > epsLLAverage):
        gmmUpd = train_GMM_EM_Iteration(X, gmm, covType=covType, psiEig=psiEig)
        llUpd = logpdf_GMM(X, gmmUpd).mean()
        llDelta = llUpd - llOld
        if verbose:
            print('GMM - it %3d - average ll %.8e' % (it, llUpd))
        gmm = gmmUpd
        llOld = llUpd
        it += 1
    if verbose:
        print('GMM - it %3d - average ll %.8e (eps = %e)' % (it, llUpd, epsLLAverage))
    return gmm

def split_GMM_LBG(gmm, alpha=0.1, verbose=True):
    gmmOut = []
    if verbose:
        print('LBG - going from %d to %d components' % (len(gmm), len(gmm) * 2))
    for (w, mu, C) in gmm:
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0] ** 0.5 * alpha
        gmmOut.append((0.5 * w, mu - d, C))
        gmmOut.append((0.5 * w, mu + d, C))
    return gmmOut

def train_GMM_LBG_EM(X, numComponents, covType, psiEig=None, epsLLAverage=1e-6, lbgAlpha=0.1, verbose=True):
    mu, C = compute_mu_C(X)
    if covType == 'diagonal':
        C = np.diag(np.diag(C))
    if psiEig is not None:
        gmm = [(1.0, mu, smooth_covariance_matrix(C, psiEig))]
    else:
        gmm = [(1.0, mu, C)]

    while len(gmm) < numComponents:
        if verbose:
            print('Average ll before LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = split_GMM_LBG(gmm, lbgAlpha, verbose=verbose)
        if verbose:
            print('Average ll after LBG: %.8e' % logpdf_GMM(X, gmm).mean())
        gmm = train_GMM_EM(X, gmm, covType=covType, psiEig=psiEig, verbose=verbose, epsLLAverage=epsLLAverage)
    return gmm
