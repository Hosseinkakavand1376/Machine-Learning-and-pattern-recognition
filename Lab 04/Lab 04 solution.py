import numpy as np
import matplotlib.pyplot as plt

def vRow(v):
    return v.reshape((1, v.size))
def vcol(v):
    return v.reshape((v.size, 1))
def Logarithmic(D, mu, C):
    M = D.shape[0]
    p = np.linalg.inv(C)
    mean = D - mu
    c_log = np.linalg.slogdet(C)
    return -0.5*M*np.log(np.pi*2)-0.5*c_log[1]-0.5*np.sum((mean.T @ p) * mean.T, axis=1)

if __name__ == '__main__':
    plt.figure()
    Xplot = np.linspace(-8, 12, 1000).reshape(-1, 1)
    mu = np.ones((1, 1)) * 1.0
    c = np.ones((1, 1)) * 2.0
    plt.plot(Xplot.ravel(), np.exp(Logarithmic(vRow(Xplot), mu, c)))
    plt.show()

    llGAU = np.load('./llGAU.npy')
    logGAU = Logarithmic(vRow(Xplot), mu, c)
    print(f"Max diff: {np.max(np.abs(llGAU - logGAU))}")

    XND = np.load('./XND.npy')
    #print(XND.shape)
    mean_XND = np.mean(XND, axis=1).reshape(-1, 1)
    print(f"Mean:\n {mean_XND}")

    # centered data
    DC = XND - mean_XND
    Covariance = np.dot(DC, DC.T)/ XND.shape[1]
    print(f"Covariance:\n {Covariance}")
    XNDGau = Logarithmic(XND, mean_XND, Covariance)
    print(f"XNDGau:\n {np.sum(XNDGau)}")

    X1D = np.load('./X1D.npy')
    mean_X1D = np.mean(X1D, axis=1).reshape(-1, 1)
    print(f"Mean:\n {mean_X1D}")
    DC_X1D = X1D - mean_X1D
    Covariance_X1D = np.dot(DC_X1D, DC_X1D.T)/ X1D.shape[1]
    print(f"Covariance:\n {Covariance_X1D}")
    X1DGau = Logarithmic(X1D, mean_X1D, Covariance_X1D)
    print(f"X1DGau:\n {np.sum(X1DGau)}")

    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    XPlot = np.linspace(-8, 12, 1000)
    plt.plot(XPlot.ravel(), np.exp(Logarithmic(vRow(XPlot), mean_X1D, Covariance_X1D)))
    plt.show()
