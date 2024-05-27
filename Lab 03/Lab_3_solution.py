import numpy as np
import matplotlib.pyplot as plt
import scipy, scipy.linalg
def mcol(v):
    return v.reshape((v.size, 1))
def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:-1]
                attrs = mcol(np.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)


def load2():
    # The dataset is already available in the sklearn library (pay attention that the library represents samples as row vectors, not column vectors - we need to transpose the data matrix)
    import sklearn.datasets
    data, label = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return data, label


def split_data(data, label, perc=(2.0 / 3.0), seed=0):
    # Split the data 2/3 for train and 1/3 for test

    n_train = int(data.shape[1] * perc)
    np.random.seed(seed)
    index = np.random.permutation(data.shape[1])
    index_train = index[:n_train]
    index_test = index[n_train:]

    data_train = data[:, index_train]
    label_train = label[index_train]
    data_test = data[:, index_test]
    label_test = label[index_test]

    return data_train, label_train, data_test, label_test
def compute_Sw_Sb(D, L):
    unique_labels = np.unique(L)
    n_features = D.shape[0]  # Number of features
    mu = D.mean(1).reshape((D.shape[0], 1))
    # Within-class scatter matrix SW
    SW = np.zeros((n_features, n_features))
    # Between-class scatter matrix SB
    SB = np.zeros((n_features, n_features))

    for label in unique_labels:
        D_i = D[:, L == label]
        mean_i = np.mean(D_i, axis=1, keepdims=True)
        SW += (D_i - mean_i).dot((D_i - mean_i).T) / (D.shape[1])
        n_i = D_i.shape[1]  # Number of samples in the current class
        mean_diff = mean_i - mu
        SB += n_i * mean_diff.dot(mean_diff.T) / D.shape[1]
    return SW, SB

def lda(D, L, m):
    SW, SB = compute_Sw_Sb(D, L)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    #print(W)
    return W
def pca(Data, n_features):
    mean = Data.mean(1).reshape((D.shape[0], 1))
    DataCentered = Data - mean
    C = ((DataCentered) @ (DataCentered).T) / float(D.shape[1])
    U, s, Vh = np.linalg.svd(C)
    new_data = U[:, :n_features]
    #print(U[:, :m])
    return new_data
if __name__ == '__main__':
    
    # Change default font size - comment to use default values
    plt.rc('font', size=16)
    plt.rc('xtick', labelsize=16)
    plt.rc('ytick', labelsize=16)

    D, L = load('iris.csv')
    #plot_hist(D, L)
    #plot_scatter(D, L)
    P = pca(D, 4)
    y = np.dot(P.T, D)
    print(P)
    LDA = lda(D, L, 2)
    z = np.dot((LDA.T), D)
    print("LDA:\n",LDA)
    class_labels = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    plt.figure(figsize=(10, 8))
    for cls in np.unique(L):
        idx = L == cls
        plt.scatter(y[0, idx], -y[1, idx], s=50, label=class_labels[cls])
    plt.title('PCA')
    plt.xlabel('Principal direction 1')
    plt.ylabel('Principal direction 2')
    plt.legend()
    plt.show()
    class_labels = {0: 'Setosa', 1: 'Versicolor', 2: 'Virginica'}
    plt.figure(figsize=(10, 8))
    for cls in np.unique(L):
        idx1 = L == cls
        plt.scatter(z[0, idx1], z[1, idx1],s=50, label=class_labels[cls])
    plt.title('LDA')
    plt.xlabel('Linear Discriminant 1')
    plt.ylabel('Linear Discriminant 2')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 8))
    for cls in np.unique(L):
        idx = L == cls
        plt.hist(y[0, idx],bins=5,edgecolor = 'black',alpha=0.3,label=class_labels[cls])
    plt.title('PCA')
    plt.xlabel('Principal direction 1')
    plt.legend()
    plt.show()
    plt.figure(figsize=(10, 8))
    for cls in np.unique(L):
        idx = L == cls
        plt.hist(z[0, idx],bins=5,edgecolor = 'black',alpha=0.3, label=class_labels[cls])
    plt.title('LDA')
    plt.xlabel('Linear Discriminator 1')
    plt.legend()
    plt.show()
