# The Python3 Bindings

- [Sklearn like API for sklearn.svm.SVC](#sklearn-like-api-for-sklearnsvmsvc)
    - [Parameters](#parameters)
    - [Attributes](#attributes)
    - [Methods](#methods)
- [Sklearn like API for sklearn.svm.SVR](#sklearn-like-api-for-sklearnsvmsvr)
    - [Parameters](#parameters)
    - [Attributes](#attributes)
    - [Methods](#methods)
- [Bindings close to our C++ API](#bindings-close-to-our-c-api)
    - [Enumerations](#enumerations)
    - [Classes and submodules](#classes-and-submodules)
        - [plssvm.Parameter](#plssvmparameter)
        - [plssvm.ClassificationDataSet and plssvm.RegressionDataSet](#plssvmclassificationdataset-and-plssvmregressiondataset)
        - [plssvm.MinMaxScaler](#plssvmminmaxscaler)
        - [plssvm.CSVC and plssvm.CSVR](#plssvmcsvc-and-plssvmcsvr)
        - [The backend C-SVCs and C-SVRs](#the-backend-c-svcs-and-c-svrs)
        - [plssvm.ClassificationModel and plssvm.RegressionModel](#plssvmclassificationmodel-and-plssvmregressionmodel)
        - [plssvm.detail.tracking.PerformanceTracker](#plssvmdetailtrackingperformancetracker)
        - [plssvm.detail.tracking.Events](#plssvmdetailtrackingevent-plssvmdetailtrackingevents)
    - [Free functions](#free-functions)
    - [Module Level Attributes](#module-level-attributes)
    - [Exceptions](#exceptions)

We currently support two kinds of Python3 bindings, one reflecting the API
of [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and [
`sklearn.svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) and one extremely closely
to our C++ API.

**Note**: this page is solely meant as an API reference and overview. For examples see the
top-level [`../../examples/`](/examples) folder.

## Sklearn like API for `sklearn.svm.SVC`

The following tables show the API provided
by [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and whether we currently
support the respective constructor parameter, class attribute, or method.
Note that the documentation is a verbose copy from the sklearn SVC page with some additional information added if our
implementation differs from the sklearn implementation.

### Parameters

The following parameters are supported
by [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) when construction a
new `SVC`:

| implementation status | parameter                                                                                  | sklearn description                                                                                                                                                                                                                                                  |
|:---------------------:|--------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  :white_check_mark:   | `C : real_type, default=1.0`                                                               | Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.                                                                                                         |
|  :white_check_mark:   | `kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'laplacian', 'chi_squared'}, default='rbf'` | Specifies the kernel type to be used in the algorithm. If none is given, 'rbf' will be used. **Note**: 'precomputed' is not supported, but 'laplacian' and 'chi_squared' are supported in addition.                                                                  |
|  :white_check_mark:   | `degree : int, default=3`                                                                  | Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.                                                                                                                                                               |
|  :white_check_mark:   | `gamma : {'scale', 'auto'} or real_type, default='scale'`                                  | Kernel coefficient for various kernel functions. **Note**: the default in PLSSVM is 'auto'.                                                                                                                                                                          |
|  :white_check_mark:   | `coef0 : real_type, default=0.0`                                                           | Independent term in kernel function. It is only significant in 'poly' or 'sigmoid'.                                                                                                                                                                                  |
|          :x:          | `shrinking : bool, default=False`                                                          | Whether to use the shrinking heuristic. **Note**: not supported and makes no sense for a LS-SVM, therefore, the default is set to `False`.                                                                                                                           |
|          :x:          | `probability : bool, default=False`                                                        | Whether to enable probability estimates.                                                                                                                                                                                                                             |
|  :white_check_mark:   | `tol : real_type, default=1e-10`                                                           | Tolerance for stopping criterion. **Note**: in PLSSVM, this is equal to the (relative) epsilon used in the CG algorithm and, therefore, other values may be necessary than for `sklearn.SVC` SVM implementation.                                                     |
|          :x:          | `cache_size : real_type, default=0`                                                        | Specify the size of the kernel cache (in MB). **Note**: not applicable in PLSSVM.                                                                                                                                                                                    |
|          :x:          | `class_weight : dict or 'balanced, default=None`                                           | Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.                                                                                                                                              |
|  :white_check_mark:   | `verbose : bool, default=False`                                                            | Enable verbose output. **Note**: if set to True, more information will be displayed than it would be the case with LIBSVM (and, therefore, `sklearn.svm.SVC`).                                                                                                       |
|  :white_check_mark:   | `max_iter : int, default=-1`                                                               | Hard limit on iterations within solver, or -1 for no limit. **Note**: if -1 is provided, at most `#data_points - 1` many CG iterations are performed.                                                                                                                |
|  :white_check_mark:   | `decision_function_shape : {'ovr', 'ovo'}, default='ovr'`                                  | Whether to return a one-vs-rest ('ovr') decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one ('ovo') decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).                     |
|          :x:          | `break_ties : bool, default=False`                                                         | If true, decision_function_shape='ovr', and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned. **Note**: PLSSVM behaves as if False was provided. |
|          :x:          | `random_state : int, RandomState instance or None, default=None`                           | Controls the pseudo random number generation for shuffling the data for probability estimates. Ignored when `probability` is False.                                                                                                                                  |

**Note**: the `plssvm.svm.SVC` automatically uses the optimal (in the sense of performance) backend and target platform, 
as they were made available during PLSSVM's build step.

### Attributes

The following attributes are supported
by [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html):

| implementation status | attribute                                                                                                                          | sklearn description                                                                                                                                                                                                                                                                                                                                       |
|:---------------------:|------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  :white_check_mark:   | `class_weight_ : ndarray of shape (n_classes,)`                                                                                    | Multipliers of parameter C for each class. Computed based on the `class_weight` parameter. **Note**: returns all `1.0` since the `class_weight` parameter is currently not supported.                                                                                                                                                                     | 
|  :white_check_mark:   | `classes_ : ndarray of shape (n_classes,)`                                                                                         | The classes labels.                                                                                                                                                                                                                                                                                                                                       |
|  :white_check_mark:   | `coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features) for ovo, ndarray of shape (n_classes, n_features) for ovr` | Weights assigned to the features when `kernel="linear"`.                                                                                                                                                                                                                                                                                                  |
|          :x:          | `dual_coef_ : ndarray of shape (n_classes -1, n_SV)`                                                                               | Dual coefficients of the support vector in the decision function, multiplied by their targets.                                                                                                                                                                                                                                                            |
|  :white_check_mark:   | `fit_status_ : int`                                                                                                                | 0 if correctly fitted, 1 otherwise (will raise warning).                                                                                                                                                                                                                                                                                                  |
|  :white_check_mark:   | `intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,) for ovo, ndarray of shape (n_classes,) for ovr`                  | Constants in decision function.                                                                                                                                                                                                                                                                                                                           |
|  :white_check_mark:   | `n_features_in_ : int`                                                                                                             | Number of features seen during `fit`.                                                                                                                                                                                                                                                                                                                     |
|  :white_check_mark:   | `feature_names_in_ : ndarray of shape (n_features_in_,)`                                                                           | Names of features seen during `fit`. Only available of the data for `fit` is provided via a Pandas DataFrame and the column names are set.                                                                                                                                                                                                                |
|  :white_check_mark:   | `n_iter_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)` for 'ovo' and ndarray of shape (n_classes,) for 'ovr'              | Number of iterations run by the optimization routine to fit the model. The shape of this attribute depends on the number of models optimized which in turn depends on the number of classes and decision function. **Note**: for 'ovr' the values correspond to the number of CG iterations necessary for each right-hand side (i.e., class) to converge. |
|  :white_check_mark:   | `support_ : ndarray of shape (n_SV)`                                                                                               | Indices of support vectors.                                                                                                                                                                                                                                                                                                                               |
|  :white_check_mark:   | `support_vectors_ : ndarray of shape (n_SV, n_features)`                                                                           | Support vectors.                                                                                                                                                                                                                                                                                                                                          |
|  :white_check_mark:   | `n_support_ : ndarray of shape (n_classes,), dtype=int32`                                                                          | Number of support vectors for each class.                                                                                                                                                                                                                                                                                                                 |
|          :x:          | `probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)`                                                                      | Parameter learned in Platt scaling when `probability=True`.                                                                                                                                                                                                                                                                                               |
|          :x:          | `probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)`                                                                      | Parameter learned in Platt scaling when `probability=True`.                                                                                                                                                                                                                                                                                               |
|  :white_check_mark:   | `shape_fit_ : tuple of int of shape (n_dimensions_of_X,)`                                                                          | Array dimensions of training vector `X`.                                                                                                                                                                                                                                                                                                                  |

### Methods

The following methods are supported
by [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html):

| implementation status | method                                  | sklearn description                                                                            |
|:---------------------:|-----------------------------------------|------------------------------------------------------------------------------------------------|
|  :white_check_mark:   | `decision_function(X)`                  | Evaluate the decision function for the samples in X.                                           |
|  :white_check_mark:   | `fit(X, y[, sample_weight])`            | Fit the SVM model according to the given training data. **Note**: without `sample_weight`.     |
|          :x:          | `get_metadata_routing()`                | Get metadata routing of this object.                                                           |
|  :white_check_mark:   | `get_params([deep])`                    | Get parameters for this estimator.                                                             |
|  :white_check_mark:   | `predict(X)`                            | Perform classification on samples in X.                                                        |
|          :x:          | `predict_log_proba(X)`                  | Compute log probabilities of possible outcomes for samples in X.                               |
|          :x:          | `predict_proba(X)`                      | Compute probabilities of possible outcomes for samples in X.                                   |
|  :white_check_mark:   | `score(X, y[, sample_weight])`          | Return the mean accuracy on the given test data and labels. **Note**: without `sample_weight`. |
|          :x:          | `set_fit_request(*[, sample_weight])`   | Request metadata passed to the `fit` method.                                                   |
|  :white_check_mark:   | `set_params(**params)`                  | Set the parameters of this estimator.                                                          |
|          :x:          | `set_score_request(*[, sample_weight])` | Request metadata passed to the `score` method.                                                 |

More detailed description of the class methods:

- `decision_function(X)`: Evaluate the decision function for the samples in X.
    - Parameters:
        - `X : array-like of shape (n_samples, n_features)`: the input samples
    - Returns:
        - `X : ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)`: the decision function of the sample for
          each class in the model. If `decision_function_shape='ovr'`, the shape is `(n_samples, n_classes)`.

- `fit(X, y[, sample_weight])`: Fit the SVM model according to the given training data.
    - Parameters:
        - `X : array_like of shape (n_samples, n_features)`: Training vectors, where `n_samples` is the number of 
          samples and `n_features` is the number of features.
        - `y : array-like of shape (n_samples,)`: Target values (class labels).
        - `sample_weight : array-like of shape (n_samples,), default=None`: Per-sample weights. Rescale C per sample.
          Higher weights force the classifier to put more emphasis on these points. **Note**: not supported
    - Returns:
        - `self : object`: Fitted estimator.

- `get_metadata_routing()`: Get metadata routing of this object.
    - Returns:
        - `routing : MetadataRequest`: A MetadataRequest encapsulating routing information.

- `get_params(deep=True)`: Get parameters for this estimator.
    - Parameters:
        - `deep : bool, default=True`: If True, will return the parameters for this estimator and contained sub-objects
          that are estimators. **Note**: not applicable, therefore, ignored.
    - Returns:
        - `params : dict`: Parameter names mapped to their values.

- `predict(X)`: Perform classification on samples in X.
    - Parameters:
        - `X : array-like of shape (n_samples, n_features)`
    - Returns:
        - `y_pred : ndarray of shape (n_samples,)`: Class labels for samples in X.

- `predict_log_proba(X)`: Compute log probabilities of possible outcomes for samples in X.
    - Parameters:
        - `X : array-like of shape (n_samples, n_features)`
    - Returns:
        - `T : ndarray of shape (n_samples, n_classes)`: Returns the log-probabilities of the sample for each class in
          the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes_.

- `predict_proba(X)`: Compute probabilities of possible outcomes for samples in X.
    - Parameters:
        - `X : array-like of shape (n_samples, n_features)`
    - Returns:
        - `T : ndarray of shape (n_samples, n_classes)`: Returns the probability of the sample for each class in the
          model. The columns correspond to the classes in sorted order, as they appear in the attribute classes_.

- `score(X, y, sample_weight=None)`: Return the mean accuracy on the given test data and labels.
    - Parameters:
        - `X : array-like of shape (n_samples, n_features)`: Test samples.
        - `y : array-like of shape (n_samples,) or (n_samples, n_outputs)`: True labels for X.
        - `sample_weightarray-like of shape (n_samples,), default=None`: Sample weights.
    - Returns:
        - `score : float`: Mean accuracy of `self.predict(X)` w.r.t. `y`.

- `set_fit_request(*, sample_weight: bool | None | str = "$UNCHANGED$") → SVC`: Request metadata passed to the fit method.
    - Parameters:
        - `sample_weight : str, True, False, or None, default=sklearn.utils.metadata_routing.UNCHANGED`: Metadata
          routing for `sample_weight` parameter in `fit`.
    - Returns:
        - `self : object`: The updated object.

- `set_params(**params)`: Set the parameters of this estimator.
    - Parameters:
        - `**params : dict`: Estimator parameters.
    - Returns:
        - `self : object`: Estimator instance.

- `set_score_request(*, sample_weight: bool | None | str = "$UNCHANGED$") → SVC`: Request metadata passed to the score
  method.
    - Parameters:
        - `sample_weightstr, True, False, or None, default=sklearn.utils.metadata_routing.UNCHANGED`: Metadata routing
          for `sample_weight` parameter in `score`.
    - Returns:
        - `self : object`: The updated object.

## Sklearn like API for `sklearn.svm.SVR`

The following tables show the API provided
by [`sklearn.svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) and whether we currently
support the respective constructor parameter, class attribute, or method.
Note that the documentation is a verbose copy from the sklearn SVR page with some additional information added if our
implementation differs from the sklearn implementation.

### Parameters

The following parameters are supported
by [`sklearn.svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html) when construction a
new `SVR`:

| implementation status | parameter                                                                                  | sklearn description                                                                                                                                                                                              |
|:---------------------:|--------------------------------------------------------------------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  :white_check_mark:   | `C : real_type, default=1.0`                                                               | Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.                                                     |
|  :white_check_mark:   | `kernel : {'linear', 'poly', 'rbf', 'sigmoid', 'laplacian', 'chi_squared'}, default='rbf'` | Specifies the kernel type to be used in the algorithm. If none is given, 'rbf' will be used. **Note**: 'precomputed' is not supported, but 'laplacian' and 'chi_squared' are supported in addition.              |
|  :white_check_mark:   | `degree : int, default=3`                                                                  | Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.                                                                                                           |
|  :white_check_mark:   | `gamma : {'scale', 'auto'} or real_type, default='scale'`                                  | Kernel coefficient for various kernel functions. **Note**: the default in PLSSVM is 'auto'.                                                                                                                      |
|  :white_check_mark:   | `coef0 : real_type, default=0.0`                                                           | Independent term in kernel function. It is only significant in 'poly' or 'sigmoid'.                                                                                                                              |
|          :x:          | `shrinking : bool, default=False`                                                          | Whether to use the shrinking heuristic. **Note**: not supported, therefore, the default is set to `False`                                                                                                        |
|  :white_check_mark:   | `tol : real_type, default=1e-10`                                                           | Tolerance for stopping criterion. **Note**: in PLSSVM, this is equal to the (relative) epsilon used in the CG algorithm and, therefore, other values may be necessary than for `sklearn.SVC` SVM implementation. |
|          :x:          | `cache_size : real_type, default=0`                                                        | Specify the size of the kernel cache (in MB). **Note**: not applicable in PLSSVM.                                                                                                                                |
|  :white_check_mark:   | `verbose : bool, default=False`                                                            | Enable verbose output. **Note**: if set to True, more information will be displayed than it would be the case with LIBSVM (and, therefore, `sklearn.svm.SVC`).                                                   |
|  :white_check_mark:   | `max_iter : int, default=-1`                                                               | Hard limit on iterations within solver, or -1 for no limit. **Note**: if -1 is provided, at most `#data_points - 1` many CG iterations are performed.                                                            |
|          :x:          | `epsilon : real_type, default=0.1`                                                         | The epsilon-tube within which no penalty is associated in the training loss function. **Note**: not applicable to PLSSVM's regression notation.                                                                  |

**Note**: the `plssvm.svm.SVR` automatically uses the optimal (in the sense of performance) backend and target platform, 
as they were made available during PLSSVM's build step.

### Attributes

The following attributes are supported
by [`sklearn.svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html):

| implementation status | attribute                                                                                                             | sklearn description                                                                                                                                                                                                                                                                                                                                       |
|:---------------------:|-----------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|          :x:          | `coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)`                                              | Weights assigned to the features when `kernel="linear"`.                                                                                                                                                                                                                                                                                                  |
|          :x:          | `dual_coef_ : ndarray of shape (n_classes -1, n_SV)`                                                                  | Dual coefficients of the support vector in the decision function, multiplied by their targets.                                                                                                                                                                                                                                                            |
|  :white_check_mark:   | `fit_status_ : int`                                                                                                   | 0 if correctly fitted, 1 otherwise (will raise warning).                                                                                                                                                                                                                                                                                                  |
|          :x:          | `intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)`                                                    | Constants in decision function.                                                                                                                                                                                                                                                                                                                           |
|  :white_check_mark:   | `n_features_in_ : int`                                                                                                | Number of features seen during `fit`.                                                                                                                                                                                                                                                                                                                     |
|          :x:          | `feature_names_in_ : ndarray of shape (n_features_in_,)`                                                              | Names of features seen during `fit`.                                                                                                                                                                                                                                                                                                                      |
|  :white_check_mark:   | `n_iter_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)` for 'ovo' and ndarray of shape (n_classes,) for 'ovr' | Number of iterations run by the optimization routine to fit the model. The shape of this attribute depends on the number of models optimized which in turn depends on the number of classes and decision function. **Note**: for 'ovr' the values correspond to the number of CG iterations necessary for each right-hand side (i.e., class) to converge. |
|  :white_check_mark:   | `support_ : ndarray of shape (n_SV)`                                                                                  | Indices of support vectors.                                                                                                                                                                                                                                                                                                                               |
|  :white_check_mark:   | `support_vectors_ : ndarray of shape (n_SV, n_features)`                                                              | Support vectors.                                                                                                                                                                                                                                                                                                                                          |
|  :white_check_mark:   | `n_support_ : ndarray of shape (n_classes,), dtype=int32`                                                             | Number of support vectors for each class.                                                                                                                                                                                                                                                                                                                 |
|          :x:          | `probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)`                                                         | Parameter learned in Platt scaling when `probability=True`.                                                                                                                                                                                                                                                                                               |
|          :x:          | `probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)`                                                         | Parameter learned in Platt scaling when `probability=True`.                                                                                                                                                                                                                                                                                               |

### Methods

The following methods are supported
by [`sklearn.svm.SVR`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html):

| implementation status | method                                  | sklearn description                                                                            |
|:---------------------:|-----------------------------------------|------------------------------------------------------------------------------------------------|
|  :white_check_mark:   | `fit(X, y[, sample_weight])`            | Fit the SVM model according to the given training data. **Note**: without `sample_weight`.     |
|          :x:          | `get_metadata_routing()`                | Get metadata routing of this object.                                                           |
|  :white_check_mark:   | `get_params([deep])`                    | Get parameters for this estimator.                                                             |
|  :white_check_mark:   | `predict(X)`                            | Perform classification on samples in X.                                                        |
|  :white_check_mark:   | `score(X, y[, sample_weight])`          | Return the mean accuracy on the given test data and labels. **Note**: without `sample_weight`. |
|          :x:          | `set_fit_request(*[, sample_weight])`   | Request metadata passed to the `fit` method.                                                   |
|  :white_check_mark:   | `set_params(**params)`                  | Set the parameters of this estimator.                                                          |
|          :x:          | `set_score_request(*[, sample_weight])` | Request metadata passed to the `score` method.                                                 |

More detailed description of the class methods:

- `fit(X, y[, sample_weight])`: Fit the SVM model according to the given training data.
    - Parameters:
        - `X : array_like of shape (n_samples, n_features) or (n_samples, n_samples)`: Training vectors,
          where `n_samples` is the number of samples and `n_features` is the number of features.
        - `y : array-like of shape (n_samples,)`: Target values (class labels).
        - `sample_weight : array-like of shape (n_samples,), default=None`: Per-sample weights. Rescale C per sample.
          Higher weights force the classifier to put more emphasis on these points. **Note**: not supported
    - Returns:
        - `self : object`: Fitted estimator.

- `get_metadata_routing()`: Get metadata routing of this object.
    - Returns:
        - `routing : MetadataRequest`: A MetadataRequest encapsulating routing information.

- `get_params(deep=True)`: Get parameters for this estimator.
    - Parameters:
        - `deep : bool, default=True`: If True, will return the parameters for this estimator and contained sub-objects
          that are estimators. **Note**: not applicable, therefore, ignored.
    - Returns:
        - `params : dict`: Parameter names mapped to their values.

- `predict(X)`: Perform classification on samples in X.
    - Parameters:
        - `X : array-like of shape (n_samples, n_features)`
    - Returns:
        - `y_pred : ndarray of shape (n_samples,)`: Class labels for samples in X.

- `score(X, y, sample_weight=None)`: Return the mean accuracy on the given test data and labels.
    - Parameters:
        - `X : array-like of shape (n_samples, n_features)`: Test samples.
        - `y : array-like of shape (n_samples,) or (n_samples, n_outputs)`: True labels for X.
        - `sample_weightarray-like of shape (n_samples,), default=None`: Sample weights.
    - Returns:
        - `score : float`: Mean accuracy of `self.predict(X)` w.r.t. `y`.

- `set_fit_request(*, sample_weight: bool | None | str = "$UNCHANGED$") → SVC`: Request metadata passed to the fit method.
    - Parameters:
        - `sample_weight : str, True, False, or None, default=sklearn.utils.metadata_routing.UNCHANGED`: Metadata
          routing for `sample_weight` parameter in `fit`.
    - Returns:
        - `self : object`: The updated object.

- `set_params(**params)`: Set the parameters of this estimator.
    - Parameters:
        - `**params : dict`: Estimator parameters.
    - Returns:
        - `self : object`: Estimator instance.

- `set_score_request(*, sample_weight: bool | None | str = "$UNCHANGED$") → SVC`: Request metadata passed to the score
  method.
    - Parameters:
        - `sample_weightstr, True, False, or None, default=sklearn.utils.metadata_routing.UNCHANGED`: Metadata routing
          for `sample_weight` parameter in `score`.
    - Returns:
        - `self : object`: The updated object.

## Bindings close to our C++ API

### Enumerations

The following table lists all PLSSVM enumerations exposed on the Python side:

| enumeration            | values                                                                           | description                                                                                                                                                                                                                                                 |
|------------------------|----------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TargetPlatform`       | `AUTOMATIC`, `CPU`, `GPU_NVIDIA`, `GPU_AMD`, `GPU_INTEL`                         | The different supported target platforms (default: `AUTOMATIC`). If `AUTOMATIC` is provided, checks for available devices in the following order: NVIDIA GPUs -> AMD GPUs -> Intel GPUs -> CPUs.                                                            |
| `SolverType`           | `AUTOMATIC`, `CG_EXPLICIT`, `CG_IMPLICIT`                                        | The different supported solver types (default: `AUTOMATIC`). If `AUTOMATIC` is provided, the used solver types depends on the available device and system memory.                                                                                           |
| `KernelFunctionType`   | `LINEAR`, `POLYNOMIAL`, `RBF`, `SIGMOID`, `LAPLACIAN`, `CHI_SQUARED`             | The different supported kernel functions (default: `RBF`).                                                                                                                                                                                                  |
| `FileFormatType`       | `LIBSVM`, `ARFF`                                                                 | The different supported file format types (default: `LIBSVM`).                                                                                                                                                                                              |
| `GammaCoefficientType` | `AUTOMATIC`, `SCALE`                                                             | The different modes for the dynamic gamma calculation (default: `AUTOMATIC`).                                                                                                                                                                               |
| `ClassificationType`   | `OAA`, `OAO`                                                                     | The different supported multi-class classification strategies (default: `LIBSVM`).                                                                                                                                                                          |
| `BackendType`          | `AUTOMATIC`, `OPENMP`, `HPX`, `STDPAR` `CUDA`, `HIP`, `OPENCL`, `SYCL`, `KOKKOS` | The different supported backends (default: `AUTOMATIC`). If `AUTOMATIC` is provided, the selected backend depends on the used target platform.                                                                                                              |
| `VerbosityLevel`       | `QUIET`, `LIBSVM`, `TIMING`, `FULL`                                              | The different supported log levels (default: `FULL`). `QUIET` means no output, `LIBSVM` output that is as conformant as possible with LIBSVM's output, `TIMING` all timing related outputs, and `FULL` everything. Can be combined via bit-wise operations. |
| `SVMType`              | `CSVC`, `CSVR`,                                                                  | The different supported C-SVM types.                                                                                                                                                                                                                        |

If a SYCL implementation is available, additional enumerations are available:

| enumeration            | values                              | description                                                                                                                                                                                                                                               |
|------------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ImplementationType`   | `AUTOMATIC`, `DPCPP`, `ADAPTIVECPP` | The different supported SYCL implementation types (default: `AUTOMATIC`). If `AUTOMATIC` is provided, determines the used SYCL implementation based on the value of `-DPLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION` provided during PLSSVM'S build step. |
| `KernelInvocationType` | `AUTOMATIC`, `ND_RANGE`             | The different supported SYCL kernel invocation types (default: `AUTOMATIC`). If `AUTOMATIC` is provided, simply uses `ND_RANGE` (only implemented to be able to add new invocation types in the future).                                                  |

If the stdpar backend is available, an additional enumeration is available:

| enumeration          | values                                                        | description                                     |
|----------------------|---------------------------------------------------------------|-------------------------------------------------|
| `ImplementationType` | `NVHPC`, `ROC_STDPAR`, `INTEL_LLVM`, `ADAPTIVECPP`, `GNU_TBB` | The different supported stdpar implementations. |

If the Kokos backend is available, an additional enumeration is available:

| enumeration      | values                                                                                 | description                                      |
|------------------|----------------------------------------------------------------------------------------|--------------------------------------------------|
| `ExecutionSpace` | `CUDA`, `HIP`, `SYCL`, `HPX`, `OPENMP`, `OPENMPTARGET`, `OPENACC`, `THREADS`, `SERIAL` | The different supported Kokkos execution spaces. |

**Note**: all our enumerations support implicit conversions from Python strings to the correct PLSSVM enumeration value.

### Classes and submodules

The following tables list all PLSSVM classes exposed on the Python side:

#### `plssvm.Parameter`

The parameter class encapsulates all necessary hyperparameters needed to fit an SVM.

| constructors                                                                                                                  | description                                                            |
|-------------------------------------------------------------------------------------------------------------------------------|------------------------------------------------------------------------|
| `Parameter(kernel_type=plssvm.KernelFunctionType.RBF, degree=3, gamma=plssvm.GammaCoefficientType.AUTO, coef0=0.0, cost=1.0)` | Construct a parameter object using the provided hyper-parameter value. |

| attributes                         | description                                                                                                                                                                                                                                                     |
|------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `kernel_type : KernelFunctionType` | The used kernel function type (default: `RBF`).                                                                                                                                                                                                                 |
| `degree : int`                     | The used degree in the polynomial kernel function (default: `3`).                                                                                                                                                                                               |
| `gamma : gamma_type`               | The used gamma in the different kernel functions (default: `AUTOMATIC`). The `gamma_type` is a `std::variant<real_type, plssvm.GammaCoefficientType`, i.e., either a normal floating point value can be provided or a `GammaCoefficientType` enumeration value. |
| `coef0 : real_type`                | The used coef0 in the polynomial and sigmoid kernel function (default: `0.0`).                                                                                                                                                                                  |
| `cost : real_type`                 | The used cost factor applied to the kernel matrix's diagonal by `1 / cost` (default: `1.0`).                                                                                                                                                                    |

| methods               | description                                                                                         |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| `equivalent(params2)` | Check whether the two parameter objects are equivalent. Same as `plssvm.equivalent(self, params2)`. |
| `param1 == param2`    | Check whether two parameter objects are identical.                                                  |
| `param1 != param2`    | Check whether two parameter objects aren't identical.                                               |
| `print(param)`        | Overload to print a `plssvm.Parameter` object displaying the used hyper-parameters.                 |

#### `plssvm.ClassificationDataSet` and `plssvm.RegressionDataSet`

A class encapsulating a used classification or regression data set.
The label types are either determined by the provided labels or if no labels are given or the data is read through a 
file, they must be explicitly stated using the `type` parameter.

The following constructors and methods are available for both the classification and regression data sets:

| constructors                                                                                                                                                            | description                                                                                                                                                                                                                                                                                                                                   |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ClassificationDataSet(filename, *, type=*the used label type*, format=*depending on the extesion of the filename*, scaler=*no scaling*, comm=*used MPI communicator*)` | Construct a new data set using the data provided in the given file. Default type: `std::string` for the ClassificationDataSet, `double` for the RegressionDataSet. Default file format: determines the file content based on its extension (.arff, everything else assumed to be a LIBSVM file). Default scaler: don't scale the data points. |
| `ClassificationDataSet(X, *, type=*the used label type*, scaler=*no scaling*, comm=*used MPI communicator*)`                                                            | Construct a new data set using the provided data directly. Default type: `std::string` for the ClassificationDataSet, `double` for the RegressionDataSet. Default scaler: don't scale the data points.                                                                                                                                        |
| `ClassificationDataSet(X, y, *, scaler=*no scaling*, comm=*used MPI communicator*)`                                                                                     | Construct a new data set using the provided data and labels directly. Default scaler: don't scale the data points.                                                                                                                                                                                                                            |

| methods                                        | description                                                                                                                                                        |
|------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `save(filename, *, format=*used file format*)` | Save the current data set to the provided file.                                                                                                                    |
| `data()`                                       | Return the data points.                                                                                                                                            |
| `has_labels()`                                 | Check whether the data set is annotated with labels.                                                                                                               |
| `labels()`                                     | Return the labels, if present.                                                                                                                                     |
| `num_data_points()`                            | Return the number of data points in the data set.                                                                                                                  |
| `num_features()`                               | Return the number of features in the data set.                                                                                                                     |
| `is_scaled()`                                  | Check whether the data points have been scaled.                                                                                                                    |
| `scaling_factors()`                            | Return the scaling factors, if the data set has been scaled.                                                                                                       |
| `communicator()`                               | Return the used MPI communicator.                                                                                                                                  |
| `print(data_set)`                              | Overload to print a data set object displaying the label type, the number of data points and features as well as the classes and scaling interval (if applicable). |

The following methods are **only** available for a `plssvm.ClassificationDataSet`:

| methods         | description                                                           |
|-----------------|-----------------------------------------------------------------------|
| `num_classes()` | Return the number of classes. **Note**: `0` if no labels are present. |
| `classes()`     | Return the different classes, if labels are present.                  |

#### `plssvm.MinMaxScaler`

A class encapsulating and performing the scaling of a data set to the provided `[lower, upper]` range.

| constructors                                                  | description                                                          |
|---------------------------------------------------------------|----------------------------------------------------------------------|
| `MinMaxScaler(lower, upper, *, comm=*used MPI communicator*)` | Scale all data points feature-wise to the interval `[lower, upper]`. |
| `MinMaxScaler(interval, *, comm=*used MPI communicator*)`     | Scale all data points feature-wise to the provided interval.         |
| `MinMaxScaler(filename, *, comm=*used MPI communicator*)`     | Read previously calculated scaling factors from the provided file.   |

| methods              | description                                                                                                       |
|----------------------|-------------------------------------------------------------------------------------------------------------------|
| `save(filename)`     | Save the current scaling factors to the provided file.                                                            |
| `scaling_interval()` | The scaling interval.                                                                                             |
| `scaling_factors())` | The calculated feature-wise scaling factors.                                                                      |
| `communicator()`     | Return the used MPI communicator.                                                                                 |
| `print(scaling)`     | Overload to print a data set scaling object object displaying the scaling interval and number of scaling factors. |

##### `plssvm.MinMaxScalerFactors`

A class encapsulating a scaling factor for a specific feature in a data set obtained by `plssvm.MinMaxScaler`.
**Note**: it shouldn't be necessary to directly use `plssvm.MinMaxScalerFactors` in user code.

| constructors                                       | description                                                                                           |
|----------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `MinMaxScalerFactors(feature_index, lower, upper)` | Construct a new scaling factor for the provided feature with the features minimum and maximum values. |

| attributes                  | description                                     |
|-----------------------------|-------------------------------------------------|
| `feature_index : size_type` | The index of the current feature.               |
| `lower : real_type`         | The minimum value of the current feature index. |
| `upper : real_type`         | The maximum value of the current feature index. |

| methods                 | description                                                                                                    |
|-------------------------|----------------------------------------------------------------------------------------------------------------|
| `print(scaling_factor)` | Overload to print a data set scaling object object displaying the feature's index, minimum, and maximum value. |

#### `plssvm.CSVC` and `plssvm.CSVR`

The main class responsible for fitting an SVM model and later predicting or scoring new data sets.
It uses either the provided backend type or the default determined one to create a PLSSVM C-SVM of the correct backend
type.
**Note**: the backend specific C-SVMs are only available if the respective backend has been enabled during PLSSVM's build
step.
These backend specific C-SVMs can also directly be used, e.g., `plssvm.CSVC(plssvm.BackendType.CUDA)` is equal
to `plssvm.cuda.CSVC` (the same also holds for all other backends).
If the most performant backend should be used, it is sufficient to use `plssvm.CSVC()` or `plssvm.CSVR()`.

The following constructors and methods are available for both classification `CSVC` and regression `CSVR`:

| constructors                                                                                                                                                                | description                                                                         |
|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `CSVC(backend, target, *, params=plssvm.Parameter, comm=*used MPI communicator*)`                                                                                           | Create a new C-SVM with the provided named arguments and `plssvm.Parameter` object. |
| `CSVC(pbackend, target, *, kernel_type=plssvm.KernelFunctionType.RBF, degree=3, gamma=plssvm.GammaCoefficientType.AUTO, coef0=0.0, cost=1.0, comm=*used MPI communicator*)` | Create a new C-SVM with the provided parameters and named arguments.                |

**Note**: if the backend type is `plssvm.BackendType.SYCL` two additional named parameters can be provided:
`sycl_implementation_type` to choose between DPC++ and AdaptiveCpp as SYCL implementations
and `sycl_kernel_invocation_type` to choose between the two different SYCL kernel invocation types.

**Note**: if the backend type is `plssvm.BackendType.HPX` or `plssvm.BackendType.Kokkos` special initialization and
finalization functions must be called.
However, this is **automatically** handled by our Python bindings on the module import and cleanup.

| methods                                                                                                                                    | description                                                                                                                                                                                                         |
|--------------------------------------------------------------------------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `get_params()`                                                                                                                             | Return the `plssvm.Parameter` that are used in the C-SVM to learn the model.                                                                                                                                        |
| `set_params(params=plssvm.Parameter)`                                                                                                      | Replace the current `plssvm.Parameter` with the provided one.                                                                                                                                                       |
| `set_params(*, kernel_type=KernelFunctionType.LINEAR, degree=3, gamma=plssvm.GammaCoefficientType.AUTO, coef=0.0, cost=1.0])`              | Replace the current `plssvm.Parameter` values with the provided named parameters.                                                                                                                                   |
| `get_target_platform()`                                                                                                                    | Return the target platform this C-SVM is running on.                                                                                                                                                                |
| `num_available_devices()`                                                                                                                  | Return the number of available devices, i.e., if the target platform represents a GPU, this function returns the number of used GPUs. Returns always 1 for CPU only backends.                                       |
| `communicator()`                                                                                                                           | Return the used MPI communicator.                                                                                                                                                                                   |
| `fit(data, *, epsilon=1e-10, classification=plssvm.ClassificatioType.OAA, solver=plssvm.SolverType.AUTOMATIC, max_iter=*#datapoints - 1*)` | Learn a LS-SVM model given the provided data points and optional parameters (the termination criterion in the CG algorithm, the classification strategy, the used solver, and the maximum number of CG iterations). |
| `predict(model, data)`                                                                                                                     | Predict the labels of the data set using the previously learned model.                                                                                                                                              |
| `score(model)`                                                                                                                             | Score the model with respect to itself returning its accuracy.                                                                                                                                                      |
| `score(model, data)`                                                                                                                       | Score the model given the provided data set returning its accuracy.                                                                                                                                                 |

**Note**: the `classification` named parameter is not allowed for the `CSVR`!

#### The backend `C-SVC`s and `C-SVR`s

These classes represent the backend specific C-SVMs:
- OpenMP: `plssvm.openmp.CSVC` and `plssvm.openmp.CSVR`
- HPX: `plssvm.hpx.CSVC` and `plssvm.hpx.CSVR`
- stdpar: `plssvm.stdpar.CSVC` and `plssvm.stdpar.CSVR`
- CUDA: `plssvm.cuda.CSVC` and `plssvm.cuda.CSVR`
- HIP: `plssvm.hip.CSVC` and `plssvm.hip.CSVR`
- OpenCL: `plssvm.opencl.CSVC` and `plssvm.opencl.CSVR`
- SYCL: `plssvm.sycl.CSVC` and `plssvm.sycl.CSVR`
- DPC++: `plssvm.dpcpp.CSVC` and `plssvm.dpcpp.CSVR`
- AdaptiveCpp: `plssvm.adaptivecpp.CSVC` and `plssvm.adaptivecpp.CSVR`
- Kokkos: `plssvm.kokkos.CSVC` and `plssvm.kokkos.CSVR`
**Note**: they are only available if the respective backend has been enabled during PLSSVM's build step.
**Note**: the `plssvm.sycl.CSVM` is equal to the respective `plssvm.dpcpp.CSVM` or `plssvm.adaptivecpp.CSVM` if only one
SYCL implementation is available or the SYCL implementation defined by `-DPLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION`
during PLSSVM's build step.
**Note**: when using `plssvm.stdpar.CSVM` together with AdaptiveCpp as stdpar implementation, currently only the CPU is
supported as target.

These classes inherit all methods from the base `plssvm.CSVC` or `plssvm.CSVR` classes.
The following constructors and methods are available for both classification `CSVC` and regression `CSVR`:

| constructors                                                                                                                                                      | description                                                                         |
|-------------------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------|
| `CSVC(target, *, params=plssvm.Parameter, comm=*used MPI communicator*)`                                                                                          | Create a new C-SVM with the provided named arguments and `plssvm.Parameter` object. |
| `CSVC(target, *, kernel_type=plssvm.KernelFunctionType.RBF, degree=3, gamma=plssvm.GammaCoefficientType.AUTO, coef0=0.0, cost=1.0, comm=*used MPI communicator*)` | Create a new C-SVM with the provided parameters and named arguments.                |

In case of the SYCL C-SVMs (`plssvm.sycl.CSVM`, `plssvm.dpcpp.CSVM`, and `plssvm.adaptivecpp.CSVM`; the same for the 
`CSVR`s), additionally, all constructors also accept the SYCL specific `sycl_kernel_invocation_type` keyword parameter.
Also, the following method is additional available for the backend specific C-SVM:

| methods                        | description                             |
|--------------------------------|-----------------------------------------|
| `get_kernel_invocation_type()` | Return the SYCL kernel invocation type. |

In case of the stdpar C-SVM (`plssvm.stdpar.CSVC` and `plssvm.stdpar.CSVR`) the following method is additional available for the backend specific
C-SVM.

| methods                     | description                                 |
|-----------------------------|---------------------------------------------|
| `get_implementation_type()` | Return the used stdpar implementation type. |

In case of the Kokkos C-SVM (`plssvm.kokkos.CSVC` and `plssvm.kokkos.CSVR`), additionally, all constructors also accept the Kokkos specific `kokkos_execution_space` keyword parameter.
Also, the following method is additional available for the backend specific C-SVM:

| methods                 | description                             |
|-------------------------|-----------------------------------------|
| `get_execution_space()` | Return the used Kokkos execution space. |

#### `plssvm.ClassificationModel` and `plssvm::RegressionModel`

A class encapsulating a model learned during a call to `plssvm.CSVC.fit()` or `plssvm::CSVR.fit()`. 

The following constructors and methods are available for both the classification and regression models:

| constructors                                                                                 | description                                                                                                                                                                |
|----------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ClassificationModel(filename, *, type=*the used label type*, comm=*used MPI communicator*)` | Construct a new model object by reading a previously learned model from a file. Default type: `std::string` for the ClassificationModel, `double` for the RegressionModel. |

| methods                 | description                                                                                                                                                             |
|-------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `save(filename)`        | Save the current model to the provided file.                                                                                                                            |
| `num_support_vectors()` | Return the number of support vectors. **Note**: for LS-SVMs this corresponds to the number of training data points.                                                     |
| `num_features()`        | Return the number of features each support vector has.                                                                                                                  |
| `get_params()`          | Return the `plssvm.Parameter` that were used to learn this model.                                                                                                       |
| `support_vectors()`     | Return the support vectors learned in this model. **Note**: for LS-SVMs this corresponds to all training data points.                                                   |
| `labels()`              | Return the labels of the support vectors.                                                                                                                               |
| `weights()`             | Return the learned weights.                                                                                                                                             |
| `rho()`                 | Return the learned bias values.                                                                                                                                         |
| `communicator()`        | Return the used MPI communicator.                                                                                                                                       |
| `print(model)`          | Overload to print a model object displaying the number of support vectors and features, as well as the learned biases and used classification strategy (if applicable). |

The following methods are **only** available for a `plssvm.ClassificationModel`:

| methods                     | description                              |
|-----------------------------|------------------------------------------|
| `num_classes()`             | Return the number of different classes.  |
| `classes()`                 | Return the different classes.            |
| `get_classification_type()` | Return the used classification strategy. |

#### `plssvm.detail.tracking.PerformanceTracker`

A submodule used to track various performance statistics like runtimes, but also the used setup and hyperparameters.
The tracked metrics can be saved to a YAML file for later post-processing.
**Note**: only available if PLSSVM was built with `-DPLSSVM_ENABLE_PERFORMANCE_TRACKING=ON`!

| function                                           | description                                                                            |
|----------------------------------------------------|----------------------------------------------------------------------------------------|
| `add_string_tracking_entry(category, name, value)` | Add a new tracking entry to the provided category with the given name and value.       |
| `add_parameter_tracking_entry(params)`             | Add a new tracking entry for the provided `plssvm.Parameter` object.                   |
| `add_event()`                                      | Add a new generic event to the tracker.                                                |
| `pause()`                                          | Pause the current performance tracking.                                                |
| `resume()`                                         | Resume performance tracking.                                                           |
| `save(filename)`                                   | Save all collected tracking information to the provided file.                          |
| `set_reference_time(time)`                         | Set a new reference time to which the relative event and samples times are calculated. |
| `get_reference_time()`                             | Get the current reference type.                                                        |
| `is_tracking()`                                    | Check whether performance tracking is currently enabled.                               |
| `get_tracking_entries()`                           | Return a dictionary that contains all previously added tracking entries.               |
| `get_events()`                                     | Return all previously recorded events.                                                 |
| `clear_tracking_entries()`                         | Remove all currently tracked entries from the performance tracker.                     |

#### `plssvm.detail.tracking.Event`, `plssvm.detail.tracking.Events`

Two rather similar classes.
**Note**: both classes are only available if PLSSVM was built with `-DPLSSVM_ENABLE_PERFORMANCE_TRACKING=ON`!

The `plssvm.detail.tracking.Event` class is a simple POD encapsulating the time point when
an event occurred and the respective event name.

| constructors              | description            |
|---------------------------|------------------------|
| `Event(time_point, name)` | Construct a new event. |

| attributes          | description                              |
|---------------------|------------------------------------------|
| `time_point : time` | The time point when this event occurred. |
| `name : string`     | The name of this event.                  |

The `plssvm.detail.tracking.Events` class stores multiple `plssvm.detail.tracking.Event`s.

| constructors | description                                 |
|--------------|---------------------------------------------|
| `Events()`   | Construct a new and empty events container. |

| methods                       | description                                                                   |
|-------------------------------|-------------------------------------------------------------------------------|
| `add_event(event)`            | Add a new event to the events list.                                           |
| `add_event(time_point, name)` | Add a new event that occurred at the provided time point with the given name. |
| `at(idx)`                     | Retrieve the event at the provided index.                                     |
| `num_events()`                | Return the number of stored events.                                           |
| `empty()`                     | Check whether currently any event has been stored/recorded.                   |
| `get_time_points()`           | Return all recorded time points.                                              |
| `get_names()`                 | Return all recorded names.                                                    |

### Free functions

The following table lists all free functions in PLSSVM directly callable via `plssvm.`.

| function                                                                    | description                                                                                                                                                                                                                                                                                       |
|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `list_available_target_platforms()`                                         | List all available target platforms (determined during PLSSVM's build step).                                                                                                                                                                                                                      |
| `determine_default_target_platform(platform_device_list)`                   | Determines the default target platform used given the available target platforms.                                                                                                                                                                                                                 |
| `kernel_function_type_to_math_string(kernel)`                               | Returns a math string of the provided kernel function.                                                                                                                                                                                                                                            |
| `linear_kernel_function(x, y)`                                              | Calculate the linear kernel function of two vectors: x'*y                                                                                                                                                                                                                                         |
| `polynomial_kernel_function(x, y, *, degree, gamma, coef0)`                 | Calculate the polynomial kernel function of two vectors: (gamma*x'*y+coef0)^degree, with degree ∊ ℤ, gamma > 0                                                                                                                                                                                    |
| `rbf_kernel_function(x, y, *, gamma)`                                       | Calculate the radial basis function kernel function of two vectors: exp(-gamma*\|x-y\|^2), with gamma > 0                                                                                                                                                                                         |
| `sigmoid_kernel_function(x, y, *, gamma, coef0)`                            | Calculate the sigmoid kernel function of two vectors: tanh(gamma*x'*y), with gamma > 0                                                                                                                                                                                                            |
| `laplacian_kernel_function(x, y, *, gamma)`                                 | Calculate the laplacian kernel function of two vectors: exp(-gamma*\|x-y\|_1), with gamma > 0                                                                                                                                                                                                     |
| `chi_squared_kernel_function(x, y, *, gamma)`                               | Calculate the chi-squared kernel function of two vectors: exp(-gamma*sum_i((x[i] - y[i])^2) / (x[i] + y[i])), with gamma > 0                                                                                                                                                                      |
| `kernel_function(x, y, *, params)`                                          | Calculate the kernel function provided in params with the additional parameters also provided in params.                                                                                                                                                                                          |
| `classification_type_to_full_string(classification)`                        | Returns the full string of the provided classification type, i.e., "one vs. all" and "one vs. one" instead of only "oaa" or "oao".                                                                                                                                                                |
| `calculate_number_of_classifiers(classification, num_classes)`              | Return the number of necessary classifiers in a multi-class setting with the provided classification strategy and number of different classes.                                                                                                                                                    |
| `list_available_backends()`                                                 | List all available backends (determined during PLSSVM's build step).                                                                                                                                                                                                                              |
| `determine_default_backend(available_backends, available_target_platforms)` | Determines the default backend used given the available backends and target platforms.                                                                                                                                                                                                            |
| `quiet()`                                                                   | Supress **all** command line output of PLSSVM functions.                                                                                                                                                                                                                                          | 
| `get_verbosity()`                                                           | Return the current verbosity level.                                                                                                                                                                                                                                                               |
| `set_verbosity(verbosity)`                                                  | Explicitly set the current verbosity level. `plssvm.set_verbosity(plssvm.VerbosityLevel.QUIET)` is equal to `plssvm.quiet()`.                                                                                                                                                                     |
| `equivalent(params1, params2)`                                              | Check whether the two parameter classes are equivalent, i.e., the parameters for **the current kernel function** are identical. E.g., for the rbf kernel function the gamma values must be identical, but the degree values can be different, since degree isn't used in the rbf kernel function. |
| `get_gamma_string(gamma)`                                                   | Returns the gamma string based on the active member in the `gamma_type` `std::variant`.                                                                                                                                                                                                           |
| `calculate_gamma_value(gamma, matrix)`                                      | Calculate the value of gamma based on the active member in the `gamma_type` `std::variant`.                                                                                                                                                                                                       |
| `list_available_svm_types()`                                                | List all available SVM types (C-SVC or C-SVR).                                                                                                                                                                                                                                                    |
| `svm_type_to_task_name(svm_type)`                                           | Returns the task name (classification or regression) associated with the provided SVM type.                                                                                                                                                                                                       |
| `svm_type_from_model_file(model_file)`                                      | Returns the SVM type used to create the provided model file.                                                                                                                                                                                                                                      |
| `regression_report(y_true, y_pred, *, force_finite, output_dict)`           | Returns a regression report similar to sklearn's [`metrics.classification_report`](https://scikit-learn.org/0.15/modules/generated/sklearn.metrics.classification_report.html) for the regression task. If `output_dict` is , returns a Python dictionary, otherwise directly returns a string.   |

If a SYCL implementation is available, additional free functions are available:

| function                                | description                                                                      |
|-----------------------------------------|----------------------------------------------------------------------------------|
| `list_available_sycl_implementations()` | List all available SYCL implementations (determined during PLSSVM's build step). |

If a stdpar implementation is available, additional free functions are available:

| function                                  | description                                                                                                                                   |
|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------|
| `list_available_stdpar_implementations()` | List all available stdpar implementations (determined during PLSSVM's build step; currently always guaranteed to be only one implementation). |

### Module Level Attributes

A few model level attributes are support and directly retrievable in the top-level `plssvm` module:

| attribute          | description                                                                               |
|--------------------|-------------------------------------------------------------------------------------------|
| `__name__`         | The name of our PLSSVM library: "PLSSVM - Parallel Least Squares Support Vector Machine". |
| `__version__`      | The current PLSSVM version as version string.                                             |
| `__version_info__` | The current PLSSVM major, minor, and patch versions as tuple.                             |

### Exceptions

The PLSSVM Python3 bindings define a few new exception types:

| exception                    | description                                                                                                                                                     |
|------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `PLSSVMError`                | Base class of all other PLSSVM specific exceptions.                                                                                                             |
| `InvalidParameterError`      | If an invalid hyper-parameter has been provided in the `plssvm.Parameter` class.                                                                                |
| `FileReaderError`            | If something went wrong while reading the requested file (possibly using memory mapped IO.)                                                                     |
| `DataSetError`               | If something related to the `plssvm.ClassificationDataSet`/`plssvm.RegressionDataSet` class(es) went wrong, e.g., wrong arguments provided to the constructors. |
| `MinMaxScalerError`          | If something related to the `plssvm.MinMaxScaler` went wrong, e.g., scaling wasn't successfully.                                                                |
| `FileNotFoundError`          | If the requested data or model file couldn't be found.                                                                                                          |
| `InvalidFileFormatError`     | If the requested data or model file are invalid, e.g., wrong LIBSVM model header.                                                                               |
| `UnsupportedBackendError`    | If an unsupported backend has been requested.                                                                                                                   |
| `UnsupportedKernelTypeError` | If an unsupported target platform has been requested.                                                                                                           |
| `GPUDevicePtrError`          | If something went wrong in one of the backend's GPU device pointers. **Note**: shouldn't occur in user code.                                                    |
| `MatrixError`                | If something went wrong in the internal matrix class. **Note**: shouldn't occur in user code.                                                                   |
| `KernelLaunchResourcesError` | If something went wrong during a kernel launch due to insufficient resources.                                                                                   |
| `ClassificationReportError`  | If something in the classification report went wrong. **Note**: shouldn't occur in user code.                                                                   |
| `RegressionReportError`      | If something in the regression report went wrong. **Note**: shouldn't occur in user code.                                                                       |
| `EnvironmentError`           | If something during the special environment initialization or finalization went wrong.                                                                          |

Depending on the available backends, additional `BackendError`s are also available (e.g., `plssvm.cuda.BackendError`).
