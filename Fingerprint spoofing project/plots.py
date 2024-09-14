import matplotlib.pyplot as plt
from utils import *
from MVG import *
def plot_histograms(D, L, class_labels):
    n_features = D.shape[0]
    for i in range(n_features):
        title = "feature" + str(i)
        plt.figure()
        plt.title(title)

        y = D[:, L == 0][i]
        plt.hist(y, bins='auto', density=True, alpha=0.4, linewidth=1.0, color='blue', edgecolor='black',
                 label=class_labels[0])
        y = D[:, L == 1][i]
        plt.hist(y, bins='auto', density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black',
                 label=class_labels[1])
        plt.legend()
        plt.savefig(f'histograms_{i}.png')
        plt.show()


def plot_pca_histograms(D, L, class_labels):
    n_features = D.shape[0]
    for i in range(n_features):
        plt.figure()
        plt.title(f"PCA Direction {i + 1}")
        plt.hist(D[i, L == 0], bins='auto', density=True, alpha=0.4, linewidth=1.0, color='blue', edgecolor='black',
                 label=class_labels[0])
        plt.hist(D[i, L == 1], bins='auto', density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black',
                 label=class_labels[1])
        plt.legend()
        plt.savefig(f'pca_histograms_{i + 1}.png')
        plt.show()


def plot_lda_histogram(DLDA, L, title, class_labels):
    plt.figure()
    plt.title(title)
    plt.hist(DLDA[0, L == 0], bins='auto', density=True, alpha=0.4, linewidth=1.0, color='blue', edgecolor='black',
             label=class_labels[0])
    plt.hist(DLDA[0, L == 1], bins='auto', density=True, alpha=0.4, linewidth=1.0, color='red', edgecolor='black',
             label=class_labels[1])
    plt.legend()
    plt.savefig(f'lda_histogram_with_{title}.png')
    plt.show()
def plot_scatter_plots_separate(D, L, class_labels):
    num_features = D.shape[0]
    for i in range(num_features):
        for j in range(i + 1, num_features):
            plt.figure()
            plt.scatter(D[i, L == 0], D[j, L == 0], label=class_labels[0], alpha=0.5,color='blue')
            plt.scatter(D[i, L == 1], D[j, L == 1], label=class_labels[1], alpha=0.5,color='red')
            plt.xlabel(f'Feature {i}')
            plt.ylabel(f'Feature {j}')
            plt.legend()
            plt.title(f'Scatter Plot of Feature {i} vs Feature {j}')
            plt.savefig(f'scatter_plots_separate_{i}_{j}.png')
            plt.show()
def plot_scatter_plots(D, L, class_labels):
    n_features = D.shape[0]
    fig, axes = plt.subplots(n_features, n_features, figsize=(20, 20))
    for i in range(n_features):
        for j in range(n_features):
            ax = axes[i, j]
            if i == j:
                ax.hist(D[:, L == 0][i], bins='auto', alpha=0.5, color='blue', label=class_labels[0], edgecolor='black',
                        density=True)
                ax.hist(D[:, L == 1][i], bins='auto', alpha=0.5, color='red', label=class_labels[1], edgecolor='black',
                        density=True)
                ax.set_ylabel(f'Feature {i}')
                ax.set_xlabel(f'Feature {i}')
                ax.set_ylim(0, 1)
                ax.legend(loc='best')
            else:
                ax.scatter(D[i, L == 0], D[j, L == 0], alpha=0.5, color='blue', label=class_labels[0], edgecolor='black')
                ax.scatter(D[i, L == 1], D[j, L == 1], alpha=0.5, color='red', label=class_labels[1], edgecolor='black')
                if i == n_features - 1:
                    ax.set_xlabel(f'Feature {j}')
                if j == 0:
                    ax.set_ylabel(f'Feature {i}')
                ax.legend(loc='best')
    plt.tight_layout()
    plt.savefig('scatter_plots.png')
    plt.show()


def plot_gaussian_fits_together(D, L, ml_estimates, class_labels):
    n_features = D.shape[0]
    fig, axes = plt.subplots(n_features, 1, figsize=(10, n_features * 5))
    colors = {0: 'blue', 1: 'red'}

    for i in range(n_features):
        ax = axes[i]
        for cls in class_labels.keys():
            feature_data = D[i, L == cls]
            ax.hist(feature_data, bins='auto', density=True, alpha=0.4, color=colors[cls], edgecolor='black',
                    label=f'{class_labels[cls]} Histogram')

            X = np.linspace(min(feature_data), max(feature_data), 1000)
            mu = ml_estimates[cls][0][i]
            sigma = ml_estimates[cls][1][i]
            ax.plot(X, np.exp(logpdf_GAU_ND(X.reshape(1, -1), mu.reshape(1, 1), np.diag([sigma]))),
                    color=colors[cls], label=f'{class_labels[cls]} Gaussian Fit')

        ax.set_title(f'Feature {i}')
        ax.legend()

    plt.tight_layout()
    plt.savefig('gaussian_fits_together.png')
    plt.show()


def plot_gaussian_fits_separate(D, L, ml_estimates, class_labels):
    n_features = D.shape[0]
    colors = {0: 'blue', 1: 'red'}
    for i in range(n_features):
        plt.figure()
        for cls in class_labels.keys():
            feature_data = D[i, L == cls]
            plt.hist(feature_data,bins='auto', density=True, alpha=0.4, color=colors[cls], edgecolor='black',
                     label=f'{class_labels[cls]} Histogram')

            X = np.linspace(min(feature_data), max(feature_data), 1000)
            mu = ml_estimates[cls][0][i]
            sigma = ml_estimates[cls][1][i]
            plt.plot(X, np.exp(logpdf_GAU_ND(X.reshape(1, -1), mu.reshape(1, 1), np.diag([sigma]))),
                     color=colors[cls], label=f'{class_labels[cls]} Gaussian Fit')

        plt.title(f'Feature {i}')
        plt.legend()
        plt.savefig(f'gaussian_fits_separate_{i}.png')
        plt.show()


def plot_correlation_matrices(D, L, class_labels):
    corr_overall = np.corrcoef(D)
    corr_class_0 = np.corrcoef(D[:, L == 0])
    corr_class_1 = np.corrcoef(D[:, L == 1])

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    cax = axes[0].matshow(corr_overall, cmap='coolwarm')
    for (i, j), val in np.ndenumerate(corr_overall):
        axes[0].text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
    axes[0].set_title('Overall Correlation Matrix')
    fig.colorbar(cax, ax=axes[0])

    cax = axes[1].matshow(corr_class_0, cmap='coolwarm')
    for (i, j), val in np.ndenumerate(corr_class_0):
        axes[1].text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
    axes[1].set_title(f'Class {class_labels[0]} Correlation Matrix')
    fig.colorbar(cax, ax=axes[1])

    cax = axes[2].matshow(corr_class_1, cmap='coolwarm')
    for (i, j), val in np.ndenumerate(corr_class_1):
        axes[2].text(j, i, f'{val:.2f}', ha='center', va='center', color='black')
    axes[2].set_title(f'Class {class_labels[1]} Correlation Matrix')
    fig.colorbar(cax, ax=axes[2])

    plt.tight_layout()
    plt.savefig('correlation_matrices.png')
    plt.show()

def plot_bayes_error(llr, labels, Cfn, Cfp, title):
    print(f'Plotting {title}')
    effPriorLogOdds = np.linspace(-4, 4, 21)
    eff_priors = 1 / (1 + np.exp(-effPriorLogOdds))
    dcf = []
    mindcf = []

    for p in eff_priors:
        normalized_dcf = compute_DCF(llr, labels, p, Cfn, Cfp)
        min_dcf = compute_minDCF(llr, labels, p, Cfn, Cfp)
        dcf.append(normalized_dcf)
        mindcf.append(min_dcf)
    fig, ax = plt.subplots()
    ax.plot(effPriorLogOdds, dcf, label='Normalized DCF', color='red')
    ax.plot(effPriorLogOdds, mindcf, label='Normalized minDCF', color='blue', linestyle='--')
    ax.set_xlabel('Effective prior probability')
    ax.set_ylabel('Detection Cost Function')
    ax.set_ylim([0, 1.1])
    ax.set_xlim([-4, 4])
    plt.title(title)
    ax.legend()
    plt.savefig(f'bayes_error_{title}_no_pca.png')
    plt.show()

def plot_confusion_matrix(conf_matrix, title):
    plt.imshow(conf_matrix, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(conf_matrix.shape[0])
    plt.xticks(tick_marks, tick_marks)
    plt.yticks(tick_marks, tick_marks)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    thresh = conf_matrix.max() / 2.
    for i, j in np.ndindex(conf_matrix.shape):
        plt.text(j, i, format(conf_matrix[i, j], 'd'),
                 horizontalalignment="center",
                 color="white" if conf_matrix[i, j] > thresh else "black")
    plt.show()


