# Toy Examples of PLSSVM using its sklearn like plssvm.SVC and plssvm.SVR Python bindings

This directory contains examples for our sklearn like `plssvm.SVC` and `plssvm.SVR` Python bindings. 
Most of the examples are copied from `https://github.com/scikit-learn/scikit-learn/tree/main/examples/svm` with 
slight changes (e.g., usage of PLSSVM's additional kernel functions). 
Some examples are more modified and others again are PLSSVM specific.  

In the following, we will show example outputs of th respecting scripts.
Note that all of these problems involve small toy examples that do not showcase the full potential of PLSSVM's performance. 
The examples serve only the purpose of showcasing the possible features our PLSSVM Python bindings provide compared to `sklearn.svm.SVC`.

## plot_classifier_comparison.py

In this example, we compare the classification result of different classifiers using four different datasets.
The classifiers are:
1. `sklearn.svm.SVC` with the **linear kernel** function and **one-vs-rest** classification
2. `sklearn.svm.SVC` with the **linear kernel** function and **one-vs-one** classification
3. `sklearn.svm.SVC` with the **rbf kernel** function and **one-vs-rest** classification
4. `sklearn.svm.SVC` with the **rbf kernel** function and **one-vs-one** classification
5. `plssvm.SVC` with the **linear kernel** function and **one-vs-rest** classification
6. `plssvm.SVC` with the **linear kernel** function and **one-vs-one** classification
7. `plssvm.SVC` with the **rbf kernel** function and **one-vs-rest** classification
8. `plssvm.SVC` with the **rbf kernel** function and **one-vs-one** classification
9. a simple Neuronal Net using `sklearn.neural_network.MLPClassifier`

The four datasets are:
1. a binary **moons** dataset created via `sklearn.datasets.make_moons` 
2. a binary **circles** dataset created via `sklearn.datasets.make_circles`
3. a binary **linearly** separable dataset created via a modified `sklearn.datasets.make_classification`
4. a dataset with **four** classes created via `sklearn.datasets.make_blobs`

<p align="center">
  <img alt="plot_classifier_comparison.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/classifier_comparison.png" width="80%">
</p>

```text
moons:
Linear SVM (ovr): 0.875
Linear SVM (ovo): 0.875
RBF SVM (ovr): 0.975
RBF SVM (ovo): 0.975
PLSSVM Linear (ovr): 0.875
PLSSVM Linear (ovo): 0.875
PLSSVM RBF (ovr): 0.975
PLSSVM RBF (ovo): 0.975
Neural Net: 0.9
Best model: RBF SVM (ovr) (0.975)

circles:
Linear SVM (ovr): 0.4
Linear SVM (ovo): 0.4
RBF SVM (ovr): 0.875
RBF SVM (ovo): 0.875
PLSSVM Linear (ovr): 0.4
PLSSVM Linear (ovo): 0.4
PLSSVM RBF (ovr): 0.875
PLSSVM RBF (ovo): 0.875
Neural Net: 0.875
Best model: RBF SVM (ovr) (0.875)

linear:
Linear SVM (ovr): 0.925
Linear SVM (ovo): 0.925
RBF SVM (ovr): 0.95
RBF SVM (ovo): 0.95
PLSSVM Linear (ovr): 0.925
PLSSVM Linear (ovo): 0.925
PLSSVM RBF (ovr): 0.95
PLSSVM RBF (ovo): 0.95
Neural Net: 0.95
Best model: RBF SVM (ovr) (0.95)

blobs:
Linear SVM (ovr): 0.45
Linear SVM (ovo): 0.45
RBF SVM (ovr): 0.8
RBF SVM (ovo): 0.8
PLSSVM Linear (ovr): 0.7
PLSSVM Linear (ovo): 0.9
PLSSVM RBF (ovr): 0.85
PLSSVM RBF (ovo): 0.8
Neural Net: 0.875
Best model: PLSSVM Linear (ovo) (0.9)
```

The results show that PLSSVM is at least on-par with `sklearn`'s SVM implementation with respect to the model accuracy 
and at best better in the multi-class case. 
Note that now hyperparameter optimizations were performed. 

## plot_decision_boundaries_via_coef_and_intercept.py

This examples shows how we can calculate the decision boundary using the model's `coef0_` and `intercept_` attributes.

<p align="center">
  <img alt="plot_decision_boundaries_via_coef_and_intercept.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/decision_boundaries_via_coef_and_intercept.png" width="80%">
</p>

```text
SVC(kernel='linear') score is 0.89
plssvm.SVC(kernel='linear') score is 0.87
```

## plot_decision_boundary_confidence.py

This examples plots the decision boundaries for an example with four classes using the one-vs-rest and one-vs-one classification strategy for `sklearn` (top) and PLSSVM (bottom). 
The darker the shading, the higher is the confidence that a sample corresponds to the respective class.

<p align="center">
  <img alt="plot_decision_boundary_confidence.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/decision_boundary_confidence.png" width="80%">
</p>

```text
Training score SVC(C=10): 0.95
Training score SVC(C=10, decision_function_shape='ovo'): 0.95
Training score plssvm.SVC(C=10.0): 0.93
Training score plssvm.SVC(C=10.0, decision_function_shape='ovo'): 0.95
```

## plot_different_classifiers.py

This example showcases the decision boundary differences when using the different supported kernel functions and classification types in PLSSVM.

<p align="center">
  <img alt="plot_different_classifiers.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/different_classifiers.png" width="80%">
</p>

```text
plssvm.SVC(kernel='linear'): 0.79
plssvm.SVC(gamma=0.7): 0.83
plssvm.SVC(gamma=auto, kernel='polynomial'): 0.81
plssvm.SVC(kernel='sigmoid'): 0.18
plssvm.SVC(gamma='auto', kernel='laplacian'): 0.84
plssvm.SVC(gamma='auto', kernel='chi_squared'): 0.79
plssvm.SVC(decision_function_shape='ovo', kernel='linear'): 0.83
plssvm.SVC(decision_function_shape='ovo', gamma=0.7): 0.83
plssvm.SVC(decision_function_shape='ovo', gamma='auto', kernel='polynomial'): 0.80
plssvm.SVC(decision_function_shape='ovo', gamma='auto', kernel='sigmoid'): 0.31
plssvm.SVC(decision_function_shape='ovo', gamma='auto', kernel='laplacian'): 0.85
plssvm.SVC(decision_function_shape='ovo', gamma='auto', kernel='chi_squared'): 0.82
```

All kernel functions except the `sigmoid` kernel work rather good in this toy example.

## plot_digit_classification.py

This example is the standard digits classification example from `sklearn` using PLSSVM as `SVC` implementation. 

<p align="center">
  <img alt="plot_digit_classification.py confusion matrix" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/digit_classification_confusion_matrix.png" width="60%">
</p>

<p align="center">
  <img alt="plot_digit_classification.py output 1" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/digit_classification_1.png" width="40%">
  <img alt="plot_digit_classification.py output 2" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/digit_classification_2.png" width="40%">
</p>

```text
Classification report for classifier plssvm.SVC(gamma=0.001):
              precision    recall  f1-score   support

           0       0.99      0.99      0.99        88
           1       0.99      0.97      0.98        91
           2       0.99      0.98      0.98        86
           3       0.96      0.88      0.92        91
           4       0.99      0.97      0.98        92
           5       0.92      0.96      0.94        91
           6       0.99      1.00      0.99        91
           7       0.95      1.00      0.97        89
           8       0.97      0.94      0.95        88
           9       0.92      0.97      0.94        92

    accuracy                           0.96       899
   macro avg       0.97      0.96      0.96       899
weighted avg       0.97      0.96      0.96       899


Confusion matrix:
[[87  0  0  0  1  0  0  0  0  0]
 [ 0 88  1  0  0  1  0  0  0  1]
 [ 1  0 84  1  0  0  0  0  0  0]
 [ 0  0  0 80  0  4  0  4  3  0]
 [ 0  0  0  0 89  0  0  0  0  3]
 [ 0  0  0  0  0 87  1  0  0  3]
 [ 0  0  0  0  0  0 91  0  0  0]
 [ 0  0  0  0  0  0  0 89  0  0]
 [ 0  1  0  1  0  1  0  1 83  1]
 [ 0  0  0  1  0  2  0  0  0 89]]
Classification report rebuilt from confusion matrix:
              precision    recall  f1-score   support

           0       0.99      0.99      0.99        88
           1       0.99      0.97      0.98        91
           2       0.99      0.98      0.98        86
           3       0.96      0.88      0.92        91
           4       0.99      0.97      0.98        92
           5       0.92      0.96      0.94        91
           6       0.99      1.00      0.99        91
           7       0.95      1.00      0.97        89
           8       0.97      0.94      0.95        88
           9       0.92      0.97      0.94        92

    accuracy                           0.96       899
   macro avg       0.97      0.96      0.96       899
weighted avg       0.97      0.96      0.96       899
```

With the same default parameters, PLSSVM also achieves a high accuracy of 96%.

## plot_face_recognition.py

This example is the standard face recognition classification example from `sklearn` using PLSSVM as `SVC` implementation.

<p align="center">
  <img alt="plot_face_recognition.py confusion matrix" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/face_recognition_confusion_matrix.png" width="60%">
</p>

<p align="center">
  <img alt="plot_face_recognition.py result prediction" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/face_recognition.png" width="48%">
  <img alt="plot_face_recognition.py eigenfaces" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/face_recognition_eigenfaces.png" width="48%">
</p>

```text
Total dataset size:
n_samples: 1288
n_features: 1850
n_classes: 7
Extracting the top 150 eigenfaces from 966 faces
done in 1.218s
Projecting the input data on the eigenfaces orthonormal basis
done in 0.023s
Fitting the classifier to the training set
done in 92.340s
Best estimator found by grid search:
plssvm.SVC(C=1937.0267553276776, decision_function_shape=ovo, gamma=0.0013770726987119488)
Predicting people's names on the test set
done in 0.053s
                   precision    recall  f1-score   support

     Ariel Sharon       0.56      0.69      0.62        13
     Colin Powell       0.80      0.88      0.84        60
  Donald Rumsfeld       0.84      0.78      0.81        27
    George W Bush       0.91      0.95      0.93       146
Gerhard Schroeder       0.95      0.76      0.84        25
      Hugo Chavez       0.90      0.60      0.72        15
       Tony Blair       0.88      0.81      0.84        36

         accuracy                           0.87       322
        macro avg       0.84      0.78      0.80       322
     weighted avg       0.87      0.87      0.87       322
```

Again, with the same default parameters, PLSSVM also achieves a high accuracy of 87%.

## plot_feature_discretization.py

This example is the standard feature discretization example from `sklearn` using PLSSVM as `SVC` implementation.

<p align="center">
  <img alt="plot_rbf_parameters.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/feature_discretization.png" width="80%">
</p>

```text
dataset 0
---------
LogisticRegression: 0.86
SVC: 0.86
KBinsDiscretizer + LogisticRegression: 0.86
KBinsDiscretizer + SVC: 0.86
GradientBoostingClassifier: 0.90
SVC: 0.94

dataset 1
---------
LogisticRegression: 0.40
SVC: 0.40
KBinsDiscretizer + LogisticRegression: 0.78
KBinsDiscretizer + SVC: 0.78
GradientBoostingClassifier: 0.84
SVC: 0.90

dataset 2
---------
LogisticRegression: 0.98
SVC: 0.96
KBinsDiscretizer + LogisticRegression: 0.94
KBinsDiscretizer + SVC: 0.94
GradientBoostingClassifier: 0.94
SVC: 0.98
```

## plot_rbf_parameters.py

This example is the standard rbf parameter example from `sklearn` using PLSSVM as `SVC` implementation.
Plotted are the decision boundaries different `gamma` and `C` parameter combinations.

<p align="center">
  <img alt="plot_rbf_parameters.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/rbf_parameters.png" width="80%">
</p>

<p align="center">
  <img alt="plot_rbf_parameters.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/rbf_parameters_accuracy.png" width="50%">
</p>

```text
The best parameters are {'C': np.float64(1.0), 'gamma': np.float64(0.1)} with a score of 0.97
```

## plot_rbf_parameters_3_classes.py

This example is similar to `plot_rbf_parameters_classes.py` but uses three classes instead of a binary classification problem. 

<p align="center">
  <img alt="plot_rbf_parameters_3_classes.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/rbf_parameters_3_classes.png" width="80%">
</p>

<p align="center">
  <img alt="plot_rbf_parameters_3_classes.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/rbf_parameters_accuracy_3_classes.png" width="50%">
</p>

```text
The best parameters are {'C': np.float64(1000000000.0), 'gamma': np.float64(0.0001)} with a score of 0.99
```

## plot_separating_hyperplane.py

A simple example plotting the separating hyperplane computed by the `SVC`.

<p align="center">
  <img alt="plot_separating_hyperplane.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/separating_hyperplane.png" width="50%">
</p>

## plot_svm_anova.py

An example showing how to use a univariate feature selection before running an `SVC` to improve the classification scores

<p align="center">
  <img alt="plot_svm_anova.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_anova.png" width="50%">
</p>

## plot_svm_kernels.py

An example showing the decision boundary of different PLSSVM kernel functions on a two-dimensional binary data set:
<p align="center">
  <img alt="plot_svm_kernels.py used data set" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_kernels_data.png" width="25%">
</p>

The decision boundaries of the different kernels look as follows:

<table>
    <tr>
        <td>
            <p align="center">
              <img alt="plot_svm_kernels.py linear kernel decision boundary" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_kernels_linear.png" width="90%">
            </p>
        </td>
        <td>
            <p align="center">
              <img alt="plot_svm_kernels.py polynomial kernel decision boundary" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_kernels_poly.png" width="90%">
            </p>
        </td>
        <td>
            <p align="center">
              <img alt="plot_svm_kernels.py rbf kernel decision boundary" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_kernels_rbf.png" width="90%">
            </p>
        </td>
    </tr>
    <tr>
        <td>
            <p align="center">
              <img alt="plot_svm_kernels.py sigmoid kernel decision boundary" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_kernels_sigmoid.png" width="90%">
            </p>
        </td>
        <td>
            <p align="center">
              <img alt="plot_svm_kernels.py laplacian kernel decision boundary" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_kernels_laplacian.png" width="90%">
            </p>
        </td>
    </tr>
</table>

<p align="center">
  <img alt="plot_svm_kernels.py xor problem" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_kernels_xor.png" width="80%">
</p>

## plot_svm_margin.py

An example showing the margins between the support vectors and the separating hyperplane.

<p align="center">
  <img alt="plot_svm_margin.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_margin.png" width="60%">
</p>

## plot_svm_regression.py

A small examples showing the different PLSSVM kernel functions for three different functions:
1. sine function
2. `1 / x`
3. irregular function

<p align="center">
  <img alt="plot_svm_regression.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/svm_regression.png" width="80%">
</p>

```text
sin:
plssvm.SVR(C=100.0, kernel='linear'): 0.5140306530553254
plssvm.SVR(C=100.0, coef0=1.0, kernel='polynomial'): 0.8272917621450797
plssvm.SVR(C=100.0, gamma=0.1): 0.8252225527358968
plssvm.SVR(C=100.0, gamma=0.1, kernel='sigmoid'): 0.8189443968198762
plssvm.SVR(C=100.0, gamma=0.1, kernel='laplacian'): 0.9458769271392272

1/x:
plssvm.SVR(C=100.0, kernel='linear'): 0.07485462701560319
plssvm.SVR(C=100.0, coef0=1.0, kernel='polynomial'): 0.16627635088519455
plssvm.SVR(C=100.0, gamma=0.1): 0.2088519475091991
plssvm.SVR(C=100.0, gamma=0.1, kernel='sigmoid'): 0.0772003452359663
plssvm.SVR(C=100.0, gamma=0.1, kernel='laplacian'): 0.9336002517574398

irregular function:
plssvm.SVR(C=100.0, kernel='linear'): 0.7581728936594311
plssvm.SVR(C=100.0, coef0=1.0, kernel='polynomial'): 0.8942520718427964
plssvm.SVR(C=100.0, gamma=0.1): 0.9176677028897581
plssvm.SVR(C=100.0, gamma=0.1, kernel='sigmoid'): 0.9177400397114979
plssvm.SVR(C=100.0, gamma=0.1, kernel='laplacian'): 0.990297601578695
```

# Real-World Examples of PLSSVM using its sklearn like plssvm.SVC and plssvm.SVR Python bindings

In this section, we want to showcase the full potential of PLSSVM using larger real-world datasets that may be too large for
`sklearn` to process in a meaningful timeframe. 
All runtimes are gathered on two NVIDIA Tesla GPUs.

## plot_SVHN.py

The Street View House Numbers (SVHN) dataset (http://ufldl.stanford.edu/housenumbers/) containing 73'257 training 
samples and 26'032 test samples with 3072 features each (32x32 RGB images) of house numbers obtained by Google Street View images.

<p align="center">
  <img alt="plot_SVHN.py confusion matrix" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/real_world/svhn_confusion_matrix.png" width="60%">
</p>

<p align="center">
  <img alt="plot_SVHN.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/real_world/svhn.png" width="80%">
</p>

```text
training dataset size: (73257, 3072)
test dataset size: (26032, 3072)
fit model in 28.31s
predict labels in 6.54s
accuracy: 0.7398970497848801

classification report:
              precision    recall  f1-score   support

         0.0       0.77      0.71      0.74      1744
         1.0       0.68      0.91      0.78      5099
         2.0       0.79      0.80      0.80      4149
         3.0       0.74      0.63      0.68      2882
         4.0       0.72      0.80      0.76      2523
         5.0       0.81      0.65      0.72      2384
         6.0       0.72      0.65      0.68      1977
         7.0       0.77      0.68      0.72      2019
         8.0       0.84      0.56      0.67      1660
         9.0       0.71      0.67      0.69      1595

    accuracy                           0.74     26032
   macro avg       0.75      0.71      0.72     26032
weighted avg       0.75      0.74      0.74     26032
```

## plot_fashion_MNIST.py

The fashion MNIST dataset (https://www.kaggle.com/datasets/zalando-research/fashionmnist) containing 56'000 training 
samples and 14'000 test samples with 784 features each (28x28 gray scale images) of Zalando's article images.

<p align="center">
  <img alt="plot_fashion_MNIST.py confusion matrix" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/real_world/fashion_mnist_confusion_matrix.png" width="60%">
</p>

<p align="center">
  <img alt="plot_fashion_MNIST.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/real_world/fashion_mnist.png" width="80%">
</p>

```text
training dataset size: (56000, 784)
test dataset size: (14000, 784)
fit model in 25.03s
predict labels in 0.96s
accuracy: 0.9103

classification report:
               precision    recall  f1-score   support

           0       0.86      0.86      0.86      1394
           1       1.00      0.98      0.99      1402
           2       0.85      0.86      0.85      1407
           3       0.90      0.94      0.92      1449
           4       0.84      0.85      0.84      1357
           5       0.98      0.97      0.98      1449
           6       0.78      0.73      0.75      1407
           7       0.95      0.97      0.96      1359
           8       0.96      0.98      0.97      1342
           9       0.97      0.97      0.97      1434

    accuracy                           0.91     14000
   macro avg       0.91      0.91      0.91     14000
weighted avg       0.91      0.91      0.91     14000
```

## plot_california_housing.py

The California Housing Prices dataset (https://www.kaggle.com/datasets/camnugent/california-housing-prices) containing
16'512 training and 4128 test samples with 8 features each from the 1990 California census.

<p align="center">
  <img alt="plot_california_housing.py output" src="https://github.com/SC-SGS/PLSSVM/raw/regression/.figures//sklearn_examples/real_world/california_housing.png" width="60%">
</p>

```
training dataset size: (16512, 8)
test dataset size: (4128, 8)
fit model in 15.48s
predict labels in 0.03s

regression report:
Explained variance score:        0.7652740783084377
Mean absolute error:             0.3803466427509043
Mean squared error:              0.3076169728804503
R^2 score:                       0.7652511712080934
Squared correlation coefficient: 0.765472789039169
```
