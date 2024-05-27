import numpy
import matplotlib.pyplot as plt

Dlist = []
Label = []
hLabels = {
    'Iris-setosa': 0 ,
    'Iris-versicolor' : 1,
    'Iris-virginica' : 2
}
with open('iris.csv', 'r')as f:
    for line in f:
        features = line.split(',')[0:-1]
        features = numpy.array([float(i) for i in features]).reshape(-1,1)
        Dlist.append(features)
        labels = line.split(',')[-1].strip()
        labels = hLabels[labels]
        Label.append(labels)

matrix= numpy.hstack(Dlist), numpy.array(Label, dtype=numpy.int32)
# print(Label)
# print(Dlist)
# print(matrix)
D, L = matrix
hFea = {
    0: 'Sepal length',
    1: 'Sepal width',
    2: 'Petal length',
    3: 'Petal width'
}
Labels = [
    'Iris-setosa',
    'Iris-versicolor',
    'Iris-virginica'
]
def plot_hist(D, L):
    plt.figure(figsize=(15, 12))

    for i in range(4):
        plt.subplot(2, 2, i + 1)
        for j in range(3):
            plt.hist(D[i, L == j], alpha=0.5, label=Labels[j], density=True)
        title = hFea[i]
        plt.title(f'{title}')
        plt.legend(loc='best')
    plt.tight_layout()
    plt.show()

## Scatter plot
def plot_scatter(D, L):
    plt.figure(figsize=(12, 12))

    num_features = D.shape[0]  # Adjusted to match feature count in D
    for i in range(num_features):
        for j in range(num_features):
            plt.subplot(num_features, num_features, i * num_features + j + 1)
            if i != j:
                for k in range(3):
                    plt.scatter(D[i, L == k], D[j, L == k], label=Labels[k])
                plt.xlabel(hFea[j])
                plt.ylabel(hFea[i])
            else:
                for k in range(3):
                    plt.hist(D[i, L == k], alpha=0.5, label=Labels[k])
                plt.xlabel(hFea[i])
                plt.ylabel('Density')

            if i == 0 and j == 0:
                plt.legend(loc='best')

    plt.tight_layout()
    plt.show()

plot_hist(D, L)
plot_scatter(D, L)

for cls in [0,1,2]:
    print('Class', cls)
    DCls = D[:, L==cls]
    mu = DCls.mean(1).reshape(DCls.shape[0], 1)
    print('Mean:')
    print(mu)
    C = ((DCls - mu) @ (DCls - mu).T) / float(DCls.shape[1])
    print('Covariance:')
    print(C)
    var = DCls.var(1)
    std = DCls.std(1)
    print('Variance:', var)
    print('Std. dev.:', std)
    print()
