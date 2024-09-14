import os
from SVM import *
from MVG import *
from logistic import *
from plots import *
from GMM import *

# --------------------------------------------------------------------------------------
# Load dataset
D, L = load_Data('trainData.txt')
# Split dataset
(DTR, LTR), (DVAL, LVAL) = split_db_2tol(D, L)
class_labels = {0: 'Fake', 1: 'Real'}
Fake = D[:, L == 0]
Real = D[:, L == 1]
mu_fake, _, _, _, _ = statistics_computations(Fake)
print(f'Mean fake class:\n {mu_fake} ')
mu_real, _, _, _, _ = statistics_computations(Real)
print(f'Mean Real class:\n {mu_real} ')

best_models = {
    'Best Regression Model': {'min_dcf': float('inf')},
    'Best SVM Model': {'min_dcf': float('inf')},
    'Best GMM Model': {'min_dcf': float('inf')}
}
effPriorLogOdds = np.linspace(-4, 4, 21)
eff_priors = 1 / (1 + np.exp(-effPriorLogOdds))
dcf_values = {'Logistic Regression': [], 'Quadratic Regression': [], 'Linear SVM': [], 'Polynomial SVM': [],
              'RBF SVM': [], 'GMM full': [], 'GMM diagonal': []}
min_dcf_values = {'Logistic Regression': [], 'Quadratic Regression': [], 'Linear SVM': [], 'Polynomial SVM': [],
                  'RBF SVM': [], 'GMM full': [], 'GMM diagonal': []}
#-----------------------------------------------------------------------------------
# Compute ML estimates for each feature
ml_estimates = compute_ml_estimates(D, L)
# Plot histograms of training data
plot_histograms(D, L, class_labels)
# Plot the Gaussian fits
plot_gaussian_fits_separate(D, L, ml_estimates, class_labels)
plot_gaussian_fits_together(D, L, ml_estimates, class_labels)
plot_scatter_plots(D, L, class_labels)
plot_scatter_plots_separate(D, L, class_labels)
# Apply PCA to both training and validation data
D_pca = pca(D, 6)
DPCA = np.dot(D_pca.T, D)
# Plot histograms for training data
plot_pca_histograms(DPCA, L, class_labels)

# Apply LDA to the entire dataset
D_LDA = lda(DPCA, L, n_features=1)
DLDA = np.dot(D_LDA.T, DPCA)
# Plot histogram for LDA direction for the entire data
plot_lda_histogram(DLDA, L, 'With LDA', class_labels)


plot_lda_histogram(D, L, 'Without LDA', class_labels)
plot_correlation_matrices(D, L, class_labels)
#---------------------------------------------------------------------------------------
# Apply PCA to both training and validation data
vector = pca(DTR, 6)
DTR_pca = np.dot(vector.T, DTR)
DVAL_pca = np.dot(vector.T, DVAL)

# Project both training and validation data
vector_lda = lda(DTR_pca, LTR, 1)
DTR_lda = np.dot(vector_lda.T, DTR_pca)
DVAL_lda = np.dot(vector_lda.T, DVAL_pca)

# Select the threshold
threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0
print('Threshold = ', threshold)
# Compute predictions for the validation data
PVAL = np.zeros(LVAL.shape, dtype=np.int32)
PVAL[DVAL_lda[0] >= threshold] = 1
PVAL[DVAL_lda[0] < threshold] = 0

# Compute the error rate
error_rate = np.mean(PVAL != LVAL)
print(f'Validation Error Rate: {error_rate * 100:.2f}%')
print(f'Accuracy is {(1 - error_rate) * 100:.2f}')

# Initial threshold based on class means
initial_threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0

# Evaluate initial threshold
initial_error_rate, initial_accuracy = evaluate_threshold(initial_threshold, DVAL_lda, LVAL)
print(f'Initial Threshold: {initial_threshold}')
print(f'Initial Validation Error Rate: {initial_error_rate * 100:.2f}%')
print(f'Initial Accuracy: {initial_accuracy / LVAL.size * 100:.2f}%')

# Test different threshold values
thresholds = np.linspace(initial_threshold - 1, initial_threshold + 1, 10)
best_threshold = initial_threshold
best_accuracy = initial_accuracy / LVAL.size * 100
best_error_rate = initial_error_rate * 100

for threshold in thresholds:
    error_rate, accuracy = evaluate_threshold(threshold, DVAL_lda, LVAL)
    accuracy_percent = accuracy / LVAL.size * 100
    print(
        f'Threshold: {threshold:.2f}, Validation Error Rate: {error_rate * 100:.2f}%, Accuracy: {accuracy_percent:.2f}%\n')

    if accuracy_percent > best_accuracy:
        best_accuracy = accuracy_percent
        best_threshold = threshold
        best_error_rate = error_rate * 100

print('LDA calculations')
print(f'Best Threshold: {best_threshold}')
print(f'Best Validation Error Rate: {best_error_rate:.2f}%')
print(f'Best Accuracy: {best_accuracy:.2f}%')

# Test different numbers of PCA dimensions
results = []
for m in range(1, DTR.shape[0] + 1):
    # Apply PCA to training data
    vector_pca = pca(DTR, m)
    DTR_pca = np.dot(vector_pca.T, DTR)
    DVAL_pca = np.dot(vector_pca.T, DVAL)

    # Apply LDA to the PCA-transformed training data
    vector_lda = lda(DTR_pca, LTR, 1)
    DTR_lda = np.dot(vector_lda.T, DTR_pca)
    DVAL_lda = np.dot(vector_lda.T, DVAL_pca)

    # Select the threshold
    threshold = (DTR_lda[0, LTR == 0].mean() + DTR_lda[0, LTR == 1].mean()) / 2.0

    # Compute predictions for the validation data
    error_rate, accuracy = evaluate_threshold(threshold, DVAL_lda, LVAL)
    accuracy_percent = accuracy / LVAL.size * 100
    results.append((m, threshold, error_rate, accuracy_percent))

# Identify the best PCA dimensions
best_result = max(results, key=lambda x: x[3])
best_m, best_threshold, best_error_rate, best_accuracy = best_result

print(f'Best PCA Dimensions: {best_m}')
print(f'Best Threshold: {best_threshold}')
print(f'Best Validation Error Rate: {best_error_rate * 100:.2f}%')
print(f'Best Accuracy: {best_accuracy:.2f}%')

# Define the applications to evaluate
applications = [
    # (0.1, 1.0, 1.0)
    (0.5, 1.0, 1.0),
    (0.9, 1.0, 1.0)
    # (0.5, 1.0, 9.0),
    # (0.5, 9.0, 1.0)
]

for pi1, Cfn, Cfp in applications:
    # Filter the dataset to include only features 0, 1, 2, and 3
    DTR_filtered = DTR[:4, :]
    DVAL_filtered = DVAL[:4, :]
    results_filtered = {}
    # Filter the dataset to include only features 0 and 1
    DTR_0_1 = DTR[:2, :]
    DVAL_0_1 = DVAL[:2, :]
    results_2 = {}
    # Filter the dataset to include only features 2 and 3
    DTR_2_3 = DTR[2:4, :]
    DVAL_2_3 = DVAL[2:4, :]
    results_3 = {}
    # Initialize a dictionary to store the results
    results = {}
    results_pca = {}
    pca_components = 6
    # Apply PCA
    pca_matrix = pca(DTR, pca_components)
    DTR_pca = np.dot(pca_matrix.T, DTR)
    DVAL_pca = np.dot(pca_matrix.T, DVAL)
    # Variables to store results for plotting
    dcf_results = {app: [] for app in applications}
    minDCF_results = {app: [] for app in applications}
    weighted_dcf_values = {app: [] for app in applications}
    weighted_minDCF_values = {app: [] for app in applications}
    quad_dcf_results = {app: [] for app in applications}
    quad_minDCF_values = {app: [] for app in applications}
    weighted_quad_dcf_values = {app: [] for app in applications}
    weighted_quad_minDCF_values = {app: [] for app in applications}
    # Variables to store results for plotting
    dcf_results_centered = {app: [] for app in applications}
    minDCF_results_centered = {app: [] for app in applications}

    DTR_expanded = expand_features(DTR)
    DTS_expanded = expand_features(DVAL)
    # Variables to store the best results
    best_results = {}
    # Variables to store the best results
    best_results_centered = {}
    # Initialize variables to track the best DCF and minDCF
    best_actual_DCF = float('inf')
    best_min_DCF = float('inf')
    best_lambda = None
    # Center the data
    mu_DTR = DTR.mean(axis=1, keepdims=True)
    DTR_centered = DTR - mu_DTR
    DTS_centered = DVAL - mu_DTR  # Use the mean of the training data for centering validation data

    # Evaluate models for each lambda and each application
    lambdas = np.logspace(-4, 2, 13)
    llr_values = {}
    print(f"pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")
    priors = {0: 1 - pi1, 1: pi1}
    costs = (Cfn, Cfp)

    # Evaluate models without PCA
    print("Evaluating MVG models without PCA")
    results_no_pca, llr_no_pca = evaluate_models(DTR, LTR, DVAL, LVAL, priors, costs)
    # Plot with no pca
    for i, entry in enumerate(llr_no_pca):
        plot_bayes_error(llr_no_pca[f"{entry}"], LVAL, Cfn, Cfp,
                         title=f"{entry} - No PCA, pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")
    results.update(results_no_pca)
    # Evaluate models with filtered features
    print("Evaluating models with filtered features")
    results_filtered_no_pca, llr_filtered_no_pca = evaluate_models(DTR_filtered, LTR, DVAL_filtered, LVAL, priors,
                                                                   costs)
    for i, entry in enumerate(llr_filtered_no_pca):
        plot_bayes_error(llr_filtered_no_pca[f"{entry}"], LVAL, Cfn, Cfp,
                         title=f"{entry} - filtered No PCA, pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")
    results_filtered.update(results_filtered_no_pca)
    print('\n')

    print("Results using features 1 and 2:")
    results_0_1, llr_0_1 = evaluate_models(DTR_0_1, LTR, DVAL_0_1, LVAL, priors, costs)
    results_2.update(results_0_1)
    print("\nResults using features 3 and 4:")
    results_2_3, llr_2_3 = evaluate_models(DTR_2_3, LTR, DVAL_2_3, LVAL, priors, costs)
    results_3.update(results_2_3)
    print('\n')
    for n_components in range(1, DTR.shape[0] + 1):
        print(f'Pi = {pi1}, Cfn = {Cfn}, Cfp = {Cfp} ')
        priors = {0: 1 - pi1, 1: pi1}
        costs = (Cfn, Cfp)
        # Perform PCA
        pca_eigenvectors = pca(DTR, n_components)
        DTR_pca = np.dot(pca_eigenvectors.T, DTR)
        DVAL_pca = np.dot(pca_eigenvectors.T, DVAL)

        # Evaluate models with PCA
        results_with_pca, llr_with_pca = evaluate_models(DTR_pca, LTR, DVAL_pca, LVAL, priors, costs)
        results_pca[n_components] = results_with_pca
        for i, entry in enumerate(llr_with_pca):
            fig, ax = plt.subplots()
            plot_bayes_error(llr_with_pca[f"{entry}"], LVAL, costs[0], costs[1],
                             title=f"{entry} (PCA {n_components} components), pi1={pi1}, Cfn={costs[0]}, Cfp={costs[1]}")
        # Print the results
        print(f"\nResults for PCA with {n_components} components:")
        for model in results_with_pca:
            print(
                f"{model} with PCA: Accuracy: {results_with_pca[model]['accuracy'] * 100:.2f}%, Normalized DCF: {results_with_pca[model]['DCF']:.4f}, Min DCF: {results_with_pca[model]['min_dcf']:.4f}\n")

    # Function to compare minimum and actual DCFs
    def compare_dcfs(results, pi1, costs):
        for model in results:
            min_dcf = results[model]['min_dcf']
            dcf = results[model]['DCF']
            print(f"{model} with pi1={pi1}: Min DCF = {min_dcf:.4f}, Actual DCF = {dcf:.4f}")


    results1, _ = evaluate_models(DTR, LTR, DVAL, LVAL, priors, costs)
    compare_dcfs(results1, pi1, costs)

    # ----------------------------------------------------------------------------------------------------------------
    print('Evaluating Regression models: ')
    print(f"\nEvaluating Logistic Regression for application: pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")
    # Initialize variables to track the best results
    best_actual_DCF = float('inf')
    best_min_DCF = float('inf')
    best_lambda = None
    best_llr = None
    for l in lambdas:
        print(f"\nEvaluating Logistic regression model with lambda = {l}")
        w, b = trainLogReg(DTR, LTR, l)
        score = np.dot(vcol(w).T, DVAL).ravel() + b
        val_pred = (score > 0) * 1
        priors = (LTR == 1).sum() / LTR.size

        # Calculate llr
        llr = score - np.log(priors / (1 - priors))
        # Compute DCF and minDCF
        actual_DCF = compute_DCF(llr, LVAL, pi1, Cfn, Cfp)
        min_DCF = compute_minDCF(llr, LVAL, pi1, Cfn, Cfp)

        # Store results
        dcf_results[(pi1, Cfn, Cfp)].append(actual_DCF)
        minDCF_results[(pi1, Cfn, Cfp)].append(min_DCF)

        # Update the best results
        if actual_DCF < best_actual_DCF:
            best_actual_DCF = actual_DCF
            best_lambda = l

        if min_DCF < best_min_DCF:
            best_min_DCF = min_DCF
            best_llr = llr

        if min_DCF < best_models['Best Regression Model']['min_dcf']:
            best_models['Best Regression Model'] = {'model': 'Logistic Regression',
                                                    'parameters': {'w': w, 'b': b, 'lambda': l}, 'min_dcf': min_DCF}

        # Calculate error rate and accuracy for the validation set
        val_accuracy = np.mean(val_pred == LVAL)
        val_error_rate = 1 - val_accuracy
        print(f"\nLogistic Regression Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"Logistic Regression Validation Error Rate: {val_error_rate * 100:.2f}%")
        print(f"Logistic Regression Actual DCF: {actual_DCF:.4f}, Min DCF: {min_DCF:.4f}")

    # Store the best results for the current application
    best_results[(pi1, Cfn, Cfp)] = {
        'best_lambda': best_lambda,
        'best_actual_DCF': best_actual_DCF,
        'best_min_DCF': best_min_DCF
    }
    llr_values['Logistic Regression'] = best_llr
    plot_bayes_error(llr_values['Logistic Regression'], LVAL, Cfn, Cfp, title='Logistic regression')
    # Print the best results for each application manually
    print("\nBest results for each application:")

    # Manually print results for specific applications
    results = best_results[(pi1, Cfn, Cfp)]
    print(f"Application: pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")
    print(f"Best lambda: {results['best_lambda']}")
    print(f"Best Actual DCF: {results['best_actual_DCF']:.4f}")
    print(f"Best Min DCF: {results['best_min_DCF']:.4f}")
    # Plot the actual DCF and minimum DCF as a function of λ for each application
    print('Plot the actual DCF and minimum DCF as a function of λ for each application')
    plt.figure(figsize=(12, 8))
    plt.plot(lambdas, dcf_results[(pi1, Cfn, Cfp)], label=f'Actual DCF - pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}', marker='o')
    plt.plot(lambdas, minDCF_results[(pi1, Cfn, Cfp)], label=f'Min DCF - pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}', marker='x')

    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('Detection Cost Function')
    plt.title('DCF vs. λ for Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'DCFplot_regression_{pi1}_{Cfn}_{Cfp}.png')
    plt.show()

    print('\nEvaluating Weighted Logistic Regression model with lambda')
    # Initialize variables to track the best results
    best_actual_DCF = float('inf')
    best_min_DCF = float('inf')
    best_lambda = None
    best_llr = None
    for l in lambdas:
        print(f"\nEvaluating weighted Logistic regression model with lambda = {l}")
        pT = 0.1
        w, b = trainWeightedLogRegBinary(DTR, LTR, l, pT)
        score = np.dot(vcol(w).T, DVAL).ravel() + b
        val_pred = (score > 0) * 1

        # Calculate llr
        llr = score - np.log(pT / (1 - pT))
        # Compute DCF and minDCF
        actual_DCF = compute_DCF(llr, LVAL, pi1, Cfn, Cfp)
        min_DCF = compute_minDCF(llr, LVAL, pi1, Cfn, Cfp)

        # Store results
        weighted_dcf_values[(pi1, Cfn, Cfp)].append(actual_DCF)
        weighted_minDCF_values[(pi1, Cfn, Cfp)].append(min_DCF)

        # Update the best results
        if actual_DCF < best_actual_DCF:
            best_actual_DCF = actual_DCF
            best_lambda = l

        if min_DCF < best_min_DCF:
            best_min_DCF = min_DCF
            best_llr = llr

        if min_DCF < best_models['Best Regression Model']['min_dcf']:
            best_models['Best Regression Model'] = {'model': 'Logistic Regression',
                                                    'parameters': {'w': w, 'b': b, 'lambda': l}, 'min_dcf': min_DCF}

        # Calculate error rate and accuracy for the validation set
        val_accuracy = np.mean(val_pred == LVAL)
        val_error_rate = 1 - val_accuracy
        print(f"\nLogistic Regression Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"Logistic Regression Validation Error Rate: {val_error_rate * 100:.2f}%")
        print(f"Logistic Regression Actual DCF: {actual_DCF:.4f}, Min DCF: {min_DCF:.4f}")

    # Store the best results for the current application
    best_results[(pi1, Cfn, Cfp)] = {
        'best_lambda': best_lambda,
        'best_actual_DCF': best_actual_DCF,
        'best_min_DCF': best_min_DCF
    }
    llr_values['Logistic Regression'] = best_llr
    plot_bayes_error(llr_values['Logistic Regression'], LVAL, Cfn, Cfp, title='Weighted Logistic regression')
    # Print the best results for each application manually
    print("\nBest results for each application:")

    # Manually print results for specific applications
    results = best_results[(pi1, Cfn, Cfp)]
    print(f"Application: pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")
    print(f"Best lambda: {results['best_lambda']}")
    print(f"Best Actual DCF: {results['best_actual_DCF']:.4f}")
    print(f"Best Min DCF: {results['best_min_DCF']:.4f}")
    # Plot the actual DCF and minimum DCF as a function of λ for each application
    print('Plot the actual DCF and minimum DCF as a function of λ for each application')
    plt.figure(figsize=(12, 8))
    plt.plot(lambdas, weighted_dcf_values[(pi1, Cfn, Cfp)], label=f'Actual DCF - pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}', marker='o')
    plt.plot(lambdas, weighted_minDCF_values[(pi1, Cfn, Cfp)], label=f'Min DCF - pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}', marker='x')

    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('Detection Cost Function')
    plt.title('DCF vs. λ for Weighted Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'DCFplot_Weighted_{pi1}_{Cfn}_{Cfp}.png')
    plt.show()

    print(f"\nEvaluating models for application: pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")

    # Initialize variables to track the best DCF and minDCF
    best_actual_DCF = float('inf')
    best_min_DCF = float('inf')
    best_lambda = None
    for l in lambdas:
        print(f"\nEvaluating Logistic regression model with lambda = {l}")
        w, b = trainLogReg(DTR_centered, LTR, l)
        score = np.dot(vcol(w).T, DTS_centered).ravel() + b
        val_pred = (score > 0) * 1
        priors = (LTR == 1).sum() / LTR.size
        # Calculate llr
        llr = score - np.log(priors / (1 - priors))

        # Compute DCF and minDCF
        actualDCF = compute_DCF(llr, LVAL, pi1, Cfn, Cfp)
        minDCF = compute_minDCF(llr, LVAL, pi1, Cfn, Cfp)

        # Store results
        dcf_results_centered[(pi1, Cfn, Cfp)].append(actualDCF)
        minDCF_results_centered[(pi1, Cfn, Cfp)].append(minDCF)

        # Update the best results
        if actualDCF < best_actual_DCF:
            best_actual_DCF = actualDCF
            best_lambda = l

        if minDCF < best_min_DCF:
            best_min_DCF = minDCF

        # Calculate error rate and accuracy for the validation set
        val_accuracy = np.mean(val_pred == LVAL)
        val_error_rate = 1 - val_accuracy
        print(f"\nLogistic Regression Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"Logistic Regression Validation Error Rate: {val_error_rate * 100:.2f}%")
        print(f"Logistic Regression Actual DCF: {actualDCF:.4f}, Min DCF: {minDCF:.4f}")

        # Store the best results for the current application
    best_results_centered[(pi1, Cfn, Cfp)] = {
        'best_lambda': best_lambda,
        'best_actual_DCF': best_actual_DCF,
        'best_min_DCF': best_min_DCF
    }
    # Manually print results for specific applications
    print("\nBest results for each application:")

    results = best_results_centered[(pi1, Cfn, Cfp)]
    print(f"Application: pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")
    print(f"Best lambda: {results['best_lambda']}")
    print(f"Best Actual DCF: {results['best_actual_DCF']:.4f}")
    print(f"Best Min DCF: {results['best_min_DCF']:.4f}")
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, dcf_results_centered[(pi1, Cfn, Cfp)], label=f'Actual DCF (pi1={pi1}), Cfn={Cfn}, Cfp={Cfp}',
             marker='o')
    plt.plot(lambdas, minDCF_results_centered[(pi1, Cfn, Cfp)], label=f'Min DCF (pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}',
             marker='x')
    plt.xscale('log')
    plt.xlabel('Lambda')
    plt.ylabel('DCF')
    plt.grid(True)
    plt.title('DCF vs Lambda (Centered Data)')
    plt.legend()
    plt.savefig(f'DCFplot_Centered_{pi1}_{Cfn}_{Cfp}.png')
    plt.show()

    # Initialize variables to track the best DCF and minDCF
    best_actual_DCF = float('inf')
    best_min_DCF = float('inf')
    best_lambda = None
    best_llr = None
    for l in lambdas:
        print(f"\nEvaluating Quadratic Logistic regression model with lambda = {l}")
        w, b = trainLogReg(DTR_expanded, LTR, l)
        score = np.dot(vcol(w).T, DTS_expanded).ravel() + b
        val_pred = (score > 0) * 1
        priors = (LTR == 1).sum() / LTR.size
        # Calculate llr
        llr = score - np.log(priors / (1 - priors))

        # Compute DCF and minDCF
        actualDCF = compute_DCF(llr, LVAL, pi1, Cfn, Cfp)
        minDCF = compute_minDCF(llr, LVAL, pi1, Cfn, Cfp)

        # Store results
        quad_dcf_results[(pi1, Cfn, Cfp)].append(actualDCF)
        quad_minDCF_values[(pi1, Cfn, Cfp)].append(minDCF)

        # Update the best results
        if actualDCF < best_actual_DCF:
            best_actual_DCF = actualDCF
            best_lambda = l

        if minDCF < best_min_DCF:
            best_min_DCF = minDCF
            best_llr = llr
        if minDCF < best_models['Best Regression Model']['min_dcf']:
            best_models['Best Regression Model'] = {'model': 'Quadratic Regression',
                                                    'parameters': {'w': w, 'b': b, 'lambda': l}, 'min_dcf': minDCF}

        # Calculate error rate and accuracy for the validation set
        val_accuracy = np.mean(val_pred == LVAL)
        val_error_rate = 1 - val_accuracy
        print(f"\nQuadratic Logistic Regression Validation Accuracy: {val_accuracy * 100:.2f}%")
        print(f"Quadratic Logistic Regression Validation Error Rate: {val_error_rate * 100:.2f}%")
        print(f"Quadratic Logistic Regression Actual DCF: {actualDCF:.4f}, Min DCF: {minDCF:.4f}")

        # Store the best results for the current application
    best_results[(pi1, Cfn, Cfp)] = {
        'best_lambda': best_lambda,
        'best_actual_DCF': best_actual_DCF,
        'best_min_DCF': best_min_DCF
    }
    llr_values['Quadratic Regression'] = best_llr
    # Manually print results for specific applications
    print("\nBest results for each application:")

    # Replace this block with your actual applications
    results = best_results[(pi1, Cfn, Cfp)]
    print(f"Application: pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}")
    print(f"Best lambda: {results['best_lambda']}")
    print(f"Best Actual DCF: {results['best_actual_DCF']:.4f}")
    print(f"Best Min DCF: {results['best_min_DCF']:.4f}")
    plot_bayes_error(llr_values['Quadratic Regression'], LVAL, Cfn, Cfp, title=f'Quadratic Logistic regression pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}')

    plt.show()
    plt.figure(figsize=(12, 8))
    plt.plot(lambdas, quad_dcf_results[(pi1, Cfn, Cfp)], label=f'Actual DCF - pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}',
             marker='^')
    plt.plot(lambdas, quad_minDCF_values[(pi1, Cfn, Cfp)], label=f'min DCF - pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}',
             marker='s')
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('Detection Cost Function')
    plt.title('DCF vs. λ for Quadratic Logistic Regression')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'DCFplot_Quadratic_{pi1}_{Cfn}_{Cfp}.png')
    plt.show()

    # -----------------------------------------------------------------------------------------------------------------
    print('Evaluating SVM models: ')
    print('\nLinear SVM Evaluation: ')
    C_values = np.logspace(-5, 0, 11)
    K = 1.0
    minDCF_values = []
    actDCF_values = []
    llr_svm_linear = []
    best_llr = None
    best_min_DCF = float('inf')
    for C in C_values:
        w, b = train_linear_svm(DTR, LTR, C, K)
        accuracy = evaluate_svm(DVAL, LVAL, w, b)
        # Compute decision values for test set
        scores = (np.dot(vRow(w), DVAL) + b).ravel()
        pred_val = (scores > 0) * 1
        error = (pred_val != LVAL).sum() / float(LVAL.size)
        print('Error rate: %.1f' % (error * 100))
        actDCF = compute_DCF(scores, LVAL, 0.1, 1.0, 1.0)
        minDCF = compute_minDCF(scores, LVAL, 0.1, 1.0, 1.0)
        print(f'DCF: {actDCF}, minDCF {minDCF}')
        actDCF_values.append(actDCF)
        minDCF_values.append(minDCF)
        llr_svm_linear.append(scores)
        if minDCF < best_min_DCF:
            best_min_DCF = minDCF
            best_llr = scores
        if minDCF < best_models['Best SVM Model']['min_dcf']:
            best_models['Best SVM Model'] = {'model': 'Linear SVM', 'parameters': {'w': w, 'b': b, 'C': C},
                                             'min_dcf': minDCF}
    plt.figure()
    plt.plot(C_values, minDCF_values, label='minDCF')
    plt.plot(C_values, actDCF_values, label='actDCF')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('DCF')
    plt.title('(Linear SVM)minDCF and actDCF vs C')
    plt.grid(True)
    plt.legend()
    plt.savefig(f'DCFplot_Linear_{pi1}_{Cfn}_{Cfp}.png')
    plt.show()
    llr_values['Linear SVM'] = best_llr
    plot_bayes_error(llr_values['Linear SVM'], LVAL, Cfn, Cfp, title=f'Linear SVM pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}')
    # --------------------------------------------------------------------
    # Kernel SVM Evaluation with Polynomial Kernel
    mindcf_values_polynomial = []
    actdcf_values_polynomial = []
    llr_svm_poly = []
    best_llr = None
    best_min_DCF = float('inf')
    eps = 0.0
    kernel = polynomial_kernel(2, 1)
    for C in C_values:
        alpha, primal_loss, dual_loss, duality_gap = train_kernel_svm(DTR, LTR, C, kernel, eps)
        predictions, scores = evaluate_kernel_svm(DTR, LTR, DVAL, alpha, kernel, eps)
        accuracy = np.mean(predictions == (LVAL * 2.0 - 1.0))
        error = (predictions != (LVAL * 2.0 - 1.0)).sum() / float(LVAL.size)
        print(
            f"Kernel SVM (Polynomial) - C={C}, eps={eps} - primal loss {primal_loss:e} - dual loss {dual_loss:e} - duality gap {duality_gap:e}")
        print('Error rate: %.1f' % (error * 100))
        print('Accuracy: %.1f' % (accuracy * 100))
        actDCF = compute_DCF(scores, LVAL, 0.1, 1.0, 1.0)
        minDCF = compute_minDCF(scores, LVAL, 0.1, 1.0, 1.0)

        actdcf_values_polynomial.append(actDCF)
        mindcf_values_polynomial.append(minDCF)
        if minDCF < best_min_DCF:
            best_min_DCF = minDCF
            best_llr = scores  # Store the best llr values for the current lambda
        if minDCF < best_models['Best SVM Model']['min_dcf']:
            best_models['Best SVM Model'] = {'model': 'Polynomial SVM', 'parameters': {'alpha': alpha, 'C': C},
                                             'min_dcf': minDCF}
    plt.figure()
    plt.plot(C_values, mindcf_values_polynomial, label='minDCF (Poly Kernel)')
    plt.plot(C_values, actdcf_values_polynomial, label='actDCF (Poly Kernel)')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('DCF')
    plt.title('minDCF and actDCF vs C (SVM Polynomial Kernel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'DCFplot_Polynomial_{pi1}_{Cfn}_{Cfp}.png')
    plt.show()
    llr_values['Polynomial SVM'] = best_llr
    plot_bayes_error(llr_values['Polynomial SVM'], LVAL, Cfn, Cfp, title=f'Polynomial SVM pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}')

    # ------------------------------------------------------
    gamma_values = [np.exp(-4), np.exp(-3), np.exp(-2), np.exp(-1)]
    eps = 1.0
    C_values = np.logspace(-3, 2, 11)
    minDCF_values_rbf = []
    actDCF_values_rbf = []
    best_minDCF_values = []  # To store the best minDCF for each gamma
    best_C_values = []  # To store the corresponding C for the best minDCF
    for gamma in gamma_values:
        minDCF_gamma = []
        actDCF_gamma = []
        best_minDCF = float('inf')
        best_C = None
        best_scores = None
        for C in C_values:
            kernel = rbf_kernel(gamma)
            alpha, primal_loss, dual_loss, duality_gap = train_kernel_svm(DTR, LTR, C, kernel, eps)
            predictions, scores = evaluate_kernel_svm(DTR, LTR, DVAL, alpha, kernel, eps)

            accuracy = np.mean(predictions == (LVAL * 2.0 - 1.0))
            error = (predictions != (LVAL * 2.0 - 1.0)).sum() / float(LVAL.size)
            print(
                f"Kernel SVM RBF Kernel - C={C:e}, gamma={gamma:e}, eps={eps} - primal loss {primal_loss:e} - dual loss {dual_loss:e} - duality gap {duality_gap:e}")
            print('Error rate: %.1f' % (error * 100))
            print('Accuracy: %.1f' % (accuracy * 100))

            actDCF = compute_DCF(scores, LVAL, 0.1, 1.0, 1.0)
            minDCF = compute_minDCF(scores, LVAL, 0.1, 1.0, 1.0)
            print(f'DCF: {actDCF}, minDCF: {minDCF}')
            actDCF_gamma.append(actDCF)
            minDCF_gamma.append(minDCF)
            if minDCF < best_minDCF:
                best_minDCF = minDCF
                best_C = C
                best_scores = scores
            if minDCF < best_models['Best SVM Model']['min_dcf']:
                best_models['Best SVM Model'] = {'model': 'RBF SVM',
                                                 'parameters': {'alpha': alpha, 'C': C, 'gamma': gamma},
                                                 'min_dcf': minDCF}
        actDCF_values_rbf.append(actDCF_gamma)
        minDCF_values_rbf.append(minDCF_gamma)
        llr_values['Kernel SVM RBF'] = best_scores

        plot_bayes_error(llr_values['Kernel SVM RBF'], LVAL, Cfn, Cfp, title=f'Kernel SVM RBF {gamma}')
    # Plotting RBF Kernel results
    plt.figure()
    for i, gamma in enumerate(gamma_values):
        plt.plot(C_values, minDCF_values_rbf[i], label=f'minDCF (RBF, gamma={gamma:.2e})')
        plt.plot(C_values, actDCF_values_rbf[i], label=f'actDCF (RBF, gamma={gamma:.2e})')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('DCF')
    plt.title('minDCF and actDCF vs C (SVM RBF Kernel)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'DCFplot_RBF_{pi1}_{Cfn}_{Cfp}.png')
    plt.show()
    
    # Kernel SVM Evaluation with Polynomial Kernel (degree 4)
    mindcf_values_polynomial_d4 = []
    actdcf_values_polynomial_d4 = []
    llr_svm_poly_d4 = None
    best_minDCF_poly_d4 = float('inf')
    eps = 0.0
    kernel = polynomial_kernel(4, 1)
    C_values = np.logspace(-5, 0, 11)

    for C in C_values:
        alpha, primal_loss, dual_loss, duality_gap = train_kernel_svm(DTR, LTR, C, kernel, eps)
        predictions, scores = evaluate_kernel_svm(DTR, LTR, DVAL, alpha, kernel, eps)
        accuracy = np.mean(predictions == (LVAL * 2.0 - 1.0))
        error = (predictions != (LVAL * 2.0 - 1.0)).sum() / float(LVAL.size)
        print(
            f"Kernel SVM (Polynomial) d=4 - C={C}, eps={eps} - primal loss {primal_loss:e} - dual loss {dual_loss:e} - duality gap {duality_gap:e}")
        print('Error rate: %.1f' % (error * 100))
        print('Accuracy: %.1f' % (accuracy * 100))
        actDCF = compute_DCF(scores, LVAL, 0.1, 1.0, 1.0)
        minDCF = compute_minDCF(scores, LVAL, 0.1, 1.0, 1.0)

        actdcf_values_polynomial_d4.append(actDCF)
        mindcf_values_polynomial_d4.append(minDCF)
        if minDCF < best_minDCF_poly_d4:
            best_minDCF_poly_d4 = minDCF
            llr_svm_poly_d4 = scores
    plot_bayes_error(llr_svm_poly_d4, LVAL, Cfn, Cfp, title=f'Kernel SVM Polynomial D4')
    plt.figure()
    plt.plot(C_values, mindcf_values_polynomial_d4, label='minDCF (Poly Kernel d=4)')
    plt.plot(C_values, actdcf_values_polynomial_d4, label='actDCF (Poly Kernel d=4)')
    plt.xscale('log')
    plt.xlabel('C (log scale)')
    plt.ylabel('DCF')
    plt.title('minDCF and actDCF vs C (SVM Polynomial Kernel d=4)')
    plt.legend()
    plt.grid(True)
    plt.savefig(f'DCFplot_Polynomial_d4_{pi1}_{Cfn}_{Cfp}.png')
    plt.show()
    # -------------------------------------------------------------------
    # Evaluate full covariance GMM models
    print("Evaluating full covariance GMM models...")
    results = {}
    llr_gmm = {'full': [],'tied': [], 'diagonal': []}
    best_llr_gmm = {'full': None,'tied': None, 'diagonal': None}  # To store the best llr values for each covType
    for covType in ['full','tied', 'diagonal']:
        results[covType] = {'minDCF': [], 'actDCF': []}
        best_minDCF = float('inf')
        best_numComponents = None
        best_scores = None
        for numComponents in [1, 2, 4, 8, 16, 32]:
            gmm0 = train_GMM_LBG_EM(DTR[:, LTR == 0], numComponents, covType=covType, psiEig=0.01)
            gmm1 = train_GMM_LBG_EM(DTR[:, LTR == 1], numComponents, covType=covType, psiEig=0.01)

            S0 = logpdf_GMM(DVAL, gmm0)
            S1 = logpdf_GMM(DVAL, gmm1)

            llr = S1 - S0
            minDCF = compute_minDCF(llr, LVAL, pi1, Cfn, Cfp)
            actDCF = compute_DCF(llr, LVAL, pi1, Cfn, Cfp)

            if minDCF < best_models['Best GMM Model']['min_dcf']:
                best_models['Best GMM Model'] = {'model': f'GMM {covType}', 'parameters': {'gmm0': gmm0, 'gmm1': gmm1,
                                                                                           'numComponents': numComponents},
                                                 'min_dcf': minDCF}
            # Update the best minDCF and corresponding numComponents for the current covType
            if minDCF < best_minDCF:
                best_minDCF = minDCF
                best_numComponents = numComponents
                llr_values[f'GMM {covType}'] = llr  # Store the best llr values for the current covType

            results[covType]['minDCF'].append(minDCF)
            results[covType]['actDCF'].append(actDCF)
            llr_gmm[covType].append(llr)
        # Store the best llr values for the current covType
        # llr_values[f'GMM {covType}'] = best_scores
        plot_bayes_error(llr_values[f'GMM {covType}'], LVAL, Cfn, Cfp,
                         title=f'Best Bayes Error for GMM {covType} pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}')
    # Print the evaluation results
    for covType in results:
        print(f"GMM Type: {covType.capitalize()}")
        for numComponents, (minDCF, actDCF) in zip([1, 2, 4, 8, 16, 32],
                                                   zip(results[covType]['minDCF'], results[covType]['actDCF'])):
            print(f"{numComponents} components - minDCF: {minDCF:.4f}, actDCF: {actDCF:.4f}")

    # # Plot the best Bayesian error for each covType
    # for covType in ['full','tied', 'diagonal']:
    #     plot_bayes_error(llr_values[f'GMM {covType}'], LVAL, Cfn, Cfp, title=f'Best Bayes Error for GMM {covType} pi1={pi1}, Cfn={Cfn}, Cfp={Cfp}')

#-------------------------------------------------------------------------------------------------
    #Save best models
    # Sort and store the best 3 models for the application
    best_models_sorted = sorted(best_models.items(), key=lambda x: x[1]['min_dcf'])[:3]

    # Create a directory for the application
    application_name = f"pi1_{pi1}_Cfn_{Cfn}_Cfp_{Cfp}"
    os.makedirs(application_name, exist_ok=True)

    # Save each model’s parameters in separate .npy files within the application directory
    for idx, (model_name, model_data) in enumerate(best_models_sorted, start=1):
        np.save(os.path.join(application_name, f'best_model_{idx}_{model_name}.npy'), model_data['parameters'])

    # Verify saved models
    for idx, (model_name, model_data) in enumerate(best_models_sorted, start=1):
        loaded_params = np.load(os.path.join(application_name, f'best_model_{idx}_{model_name}.npy'), allow_pickle=True).item()
        print(f"Model {idx} ({model_name}) for application {application_name} parameters:")
        print(loaded_params)
