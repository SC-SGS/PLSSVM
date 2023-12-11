# The Python3 Bindings

- [Sklearn like API](#sklearn-like-api)
  - [Parameters](#parameters)
  - [Attributes](#attributes)
  - [Methods](#methods)
- [Bindings close to our C++ API](#bindings-close-to-our-c-api)
  - [Enumerations](#enumerations)
  - [Classes and submodules](#classes-and-submodules)
    - [plssvm.Parameter](#plssvmparameter)
    - [plssvm.DataSet](#plssvmdataset)
    - [plssvm.CSVM](#plssvmcsvm)
    - [plssvm.openmp.CSVM, plssvm.cuda.CSVM, plssvm.hip.CSVM, plssvm.opencl.CSVM, plssvm.sycl.CSVM, plssvm.dpcpp.CSVM, plssvm.adaptivecpp.CSVM](#plssvmopenmpcsvm-plssvmcudacsvm-plssvmhipcsvm-plssvmopenclcsvm-plssvmsyclcsvm-plssvmdpcppcsvm-plssvmadaptivecppcsvm)
    - [plssvm.Model](#plssvmmodel)
    - [plssvm.Version](#plssvmversion)
    - [plssvm.detail.PerformanceTracker](#plssvmdetailperformancetracker)
  - [Free functions](#free-functions)
  - [Exceptions](#exceptions)

We currently support two kinds of Python3 bindings, one reflecting the API of [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and one extremely closely to our C++ API.

**Note**: this page is solely meant as an API reference and overview. For examples see the top-level [`../../examples/`](/examples) folder.

## Sklearn like API

The following tables show the API provided by [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) and whether we currently support the respective constructor parameter, class attribute, or method.
Note that the documentation is a verbose copy from the sklearn SVC page with some additional information added if our implementation differs from the sklearn implementation.

### Parameters

The following parameters are supported by [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html) when construction a new `SVC`:

| implementation status | parameter                                                        | sklearn description                                                                                                                                                                                                                                                      |
|:---------------------:|------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  :white_check_mark:   | `C : real_type, default=1.0`                                     | Regularization parameter. The strength of the regularization is inversely proportional to C. Must be strictly positive. The penalty is a squared l2 penalty.                                                                                                             |
|  :white_check_mark:   | `kernel : {'linear', 'poly', 'rbf'}, default='rbf'`              | Specifies the kernel type to be used in the algorithm. If none is given, 'rbf' will be used. **Note**: only 'linear', 'poly', and 'rbf' are currently supported.                                                                                                         |
|  :white_check_mark:   | `degree : int, default=3`                                        | Degree of the polynomial kernel function (‘poly’). Must be non-negative. Ignored by all other kernels.                                                                                                                                                                   |
|  :white_check_mark:   | `gamma : {'auto'} or real_type, default='auto'`                  | Kernel coefficient for 'rbf' and 'poly'. **Note**: `scale` is currently not supported, therefore, the default is set to `'auto'`.                                                                                                                                        |
|  :white_check_mark:   | `coef0 : real_type, default=0.0`                                 | Independent term in kernel function. It is only significant in 'poly'.                                                                                                                                                                                                   |
|          :x:          | `shrinking : bool, default=False`                                | Whether to use the shrinking heuristic. **Note**: not supported, therefore, the default is set to `False`                                                                                                                                                                |
|          :x:          | `probability : bool, default=False`                              | Whether to enable probability estimates.                                                                                                                                                                                                                                 |
|  :white_check_mark:   | `tol : real_type, default=1e-3`                                  | Tolerance for stopping criterion. **Note**: in PLSSVM, this is equal to the (relative) epsilon used in the CG algorithm.                                                                                                                                                 |
|          :x:          | `cache_size : real_type, default=0`                              | Specify the size of the kernel cache (in MB). **Note**: not applicable in PLSSVM.                                                                                                                                                                                        |
|          :x:          | `class_weight : dict or 'balanced, default=None`                 | Set the parameter C of class i to class_weight[i]*C for SVC. If not given, all classes are supposed to have weight one.                                                                                                                                                  |
|  :white_check_mark:   | `verbose : bool, default=False`                                  | Enable verbose output. **Note**: if set to True, more information will be displayed than it would be the case with LIBSVM (and, therefore, `sklearn.svm.SVC`).                                                                                                           |
|  :white_check_mark:   | `max_iter : int, default=-1`                                     | Hard limit on iterations within solver, or -1 for no limit. **Note**: if -1 is provided, at most `#data_points - 1` many CG iterations are performed.                                                                                                                    |
|  :white_check_mark:   | `decision_function_shape : {'ovr', 'ovo'}, default='ovr'`        | Whether to return a one-vs-rest ('ovr') decision function of shape (n_samples, n_classes) as all other classifiers, or the original one-vs-one ('ovo') decision function of libsvm which has shape (n_samples, n_classes * (n_classes - 1) / 2).                         |
|          :x:          | `break_ties : bool, default=False`                               | If true, `decision_function_shape='ovr'`, and number of classes > 2, predict will break ties according to the confidence values of decision_function; otherwise the first class among the tied classes is returned. **Note**: PLSSVM behaves as if `False` was provided. |
|          :x:          | `random_state : int, RandomState instance or None, default=None` | Controls the pseudo random number generation for shuffling the data for probability estimates. Ignored when `probability` is False.                                                                                                                                      |

**Note**: the `plssvm.SVC` automatically uses the optimal (in the sense of performance) backend and target platform, as they were made available during PLSSVM's build step.

### Attributes

The following attributes are supported by [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html):

| implementation status | attribute                                                                | sklearn description                                                                                                                                                                                                                                                                                                  |
|:---------------------:|--------------------------------------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
|  :white_check_mark:   | `class_weight_ : ndarray of shape (n_classes,)`                          | Multipliers of parameter C for each class. Computed based on the `class_weight` parameter. **Note**: returns all `1.0` since the `class_weight` parameter is currently not supported.                                                                                                                                | 
|  :white_check_mark:   | `classes_ : ndarray of shape (n_classes,)`                               | The classes labels.                                                                                                                                                                                                                                                                                                  |
|          :x:          | `coef_ : ndarray of shape (n_classes * (n_classes - 1) / 2, n_features)` | Weights assigned to the features when `kernel="linear"`.                                                                                                                                                                                                                                                             |
|          :x:          | `dual_coef_ : ndarray of shape (n_classes -1, n_SV)`                     | Dual coefficients of the support vector in the decision function, multiplied by their targets.                                                                                                                                                                                                                       |
|  :white_check_mark:   | `fit_status_ : int`                                                      | 0 if correctly fitted, 1 otherwise (will raise warning).                                                                                                                                                                                                                                                             |
|          :x:          | `intercept_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)`       | Constants in decision function.                                                                                                                                                                                                                                                                                      |
|  :white_check_mark:   | `n_features_in_ : int`                                                   | Number of features seen during `fit`.                                                                                                                                                                                                                                                                                |
|          :x:          | `feature_names_in_ : ndarray of shape (n_features_in_,)`                 | Names of features seen during `fit`.                                                                                                                                                                                                                                                                                 |
|  :white_check_mark:   | `n_iter_ : ndarray of shape (n_classes * (n_classes - 1) / 2,)`          | Number of iterations run by the optimization routine to fit the model. The shape of this attribute depends on the number of models optimized which in turn depends on the number of classes. **Note**: corresponds to the number of CG iterations, for 'ovr' all values in the array are guaranteed to be identical. |
|  :white_check_mark:   | `support_ : ndarray of shape (n_SV)`                                     | Indices of support vectors.                                                                                                                                                                                                                                                                                          |
|  :white_check_mark:   | `support_vectors_ : ndarray of shape (n_SV, n_features)`                 | Support vectors.                                                                                                                                                                                                                                                                                                     |
|  :white_check_mark:   | `n_support_ : ndarray of shape (n_classes,), dtype=int32`                | Number of support vectors for each class.                                                                                                                                                                                                                                                                            |
|          :x:          | `probA_ : ndarray of shape (n_classes * (n_classes - 1) / 2)`            | Parameter learned in Platt scaling when `probability=True`.                                                                                                                                                                                                                                                          |
|          :x:          | `probB_ : ndarray of shape (n_classes * (n_classes - 1) / 2)`            | Parameter learned in Platt scaling when `probability=True`.                                                                                                                                                                                                                                                          |
|  :white_check_mark:   | `shape_fit_ : tuple of int of shape (n_dimensions_of_X,)`                | Array dimensions of training vector `X`.                                                                                                                                                                                                                                                                             |

### Methods

The following methods are supported by [`sklearn.svm.SVC`](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html):

| implementation status | method                                  | sklearn description                                                                            |
|:---------------------:|-----------------------------------------|------------------------------------------------------------------------------------------------|
|          :x:          | `decision_function(X)`                  | Evaluate the decision function for the samples in X.                                           |
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
    - `X : ndarray of shape (n_samples, n_classes * (n_classes-1) / 2)`: the decision function of the sample for each class in the model. If `decision_function_shape='ovr'`, the shape is `(n_samples, n_classes)`.

- `fit(X, y[, sample_weight])`: Fit the SVM model according to the given training data.
  - Parameters:
    - `X : array_like of shape (n_samples, n_features) or (n_samples, n_samples)`: Training vectors, where `n_samples` is the number of samples and `n_features` is the number of features.
    - `y : array-like of shape (n_samples,)`: Target values (class labels).
    - `sample_weight : array-like of shape (n_samples,), default=None`: Per-sample weights. Rescale C per sample. Higher weights force the classifier to put more emphasis on these points. **Note**: not supported
  - Returns:
    - `self : object`: Fitted estimator.

- `get_metadata_routing()`: Get metadata routing of this object.
  - Returns:
    - `routing : MetadataRequest`: A MetadataRequest encapsulating routing information.

- `get_params(deep=True)`: Get parameters for this estimator.
  - Parameters:
    - `deep : bool, default=True`: If True, will return the parameters for this estimator and contained subobjects that are estimators. **Note**: not applicable, therefore, ignored.
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
    - `T : ndarray of shape (n_samples, n_classes)`: Returns the log-probabilities of the sample for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes_.

- `predict_proba(X)`: Compute probabilities of possible outcomes for samples in X.
  - Parameters: 
    - `X : array-like of shape (n_samples, n_features)`
  - Returns:
    - `T : ndarray of shape (n_samples, n_classes)`: Returns the probability of the sample for each class in the model. The columns correspond to the classes in sorted order, as they appear in the attribute classes_.

- `score(X, y, sample_weight=None)`: Return the mean accuracy on the given test data and labels.
  - Parameters:
    - `X : array-like of shape (n_samples, n_features)`: Test samples.
    - `y : array-like of shape (n_samples,) or (n_samples, n_outputs)`: True labels for X.
    - `sample_weightarray-like of shape (n_samples,), default=None`: Sample weights.
  - Returns:
    - `score : float`: Mean accuracy of `self.predict(X)` w.r.t. `y`.

- `set_fit_request(*, sample_weight: bool | None | str = '$UNCHANGED$') → SVC`: Request metadata passed to the fit method.
  - Parameters:
    - `sample_weight : str, True, False, or None, default=sklearn.utils.metadata_routing.UNCHANGED`: Metadata routing for `sample_weight` parameter in `fit`.
  - Returns:
    - `self : object`: The updated object.

- `set_params(**params)`: Set the parameters of this estimator.
  - Parameters:
    - `**params : dict`: Estimator parameters.
  - Returns:
    - `self : object`: Estimator instance.

- `set_score_request(*, sample_weight: bool | None | str = '$UNCHANGED$') → SVC`: Request metadata passed to the score method.
  - Parameters:
    - `sample_weightstr, True, False, or None, default=sklearn.utils.metadata_routing.UNCHANGED`: Metadata routing for `sample_weight` parameter in `score`.
  - Returns:
    - `self : object`: The updated object.

## Bindings close to our C++ API

### Enumerations

The following table lists all PLSSVM enumerations exposed on the Python side:

| enumeration          | values                                                    | description                                                                                                                                                                                                                                                 |
|----------------------|-----------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `TargetPlatform`     | `AUTOMATIC`, `CPU`, `GPU_NVIDIA`, `GPU_AMD`, `GPU_INTEL`  | The different supported target platforms (default: `AUTOMATIC`). If `AUTOMATIC` is provided, checks for available devices in the following order: NVIDIA GPUs -> AMD GPUs -> Intel GPUs -> CPUs.                                                            |
| `SolverType`         | `AUTOMATIC`, `CG_EXPLICIT`, `CG_STREAMING`, `CG_IMPLICIT` | The different supported solver types (default: `AUTOMATIC`). If `AUTOMATIC` is provided, the used solver types depends on the available device and system memory.                                                                                           |
| `KernelFunctionType` | `LINEAR`, `POLYNOMIAL`, `RBF`                             | The different supported kernel functions (default: `LINEAR`).                                                                                                                                                                                               |
| `FileFormatType`     | `LIBSVM`, `ARFF`                                          | The different supported file format types (default: `LIBSVM`).                                                                                                                                                                                              |
| `ClassificationType` | `OAA`, `OAO`                                              | The different supported multi-class classification strategies (default: `LIBSVM`).                                                                                                                                                                          |
| `BackendType`        | `AUTOMATIC`, `OPENMP`, `CUDA`, `HIP`, `OPENCL`, `SYCL`    | The different supported backends (default: `AUTOMATIC`). If `AUTOMATIC` is provided, the selected backend depends on the used target platform.                                                                                                              |
| `VerbosityLevel`     | `QUIET`, `LIBSVM`, `TIMING`, `FULL`                       | The different supported log levels (default: `FULL`). `QUIET` means no output, `LIBSVM` output that is as conformant as possible with LIBSVM's output, `TIMING` all timing related outputs, and `FULL` everything. Can be combined via bit-wise operations. |

If a SYCL implementation is available, additional enumerations are available:

| enumeration            | values                              | description                                                                                                                                                                                                                                               |
|------------------------|-------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `ImplementationType`   | `AUTOMATIC`, `DPCPP`, `ADAPTIVECPP` | The different supported SYCL implementation types (default: `AUTOMATIC`). If `AUTOMATIC` is provided, determines the used SYCL implementation based on the value of `-DPLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION` provided during PLSSVM'S build step. |
| `KernelInvocationType` | `AUTOMATIC`, `ND_RANGE`             | The different supported SYCL kernel invocation types (default: `AUTOMATIC`). If `AUTOMATIC` is provided, simply uses `ND_RANGE` (only implemented to be able to add new invocation types in the future).                                                  |

### Classes and submodules

The following tables list all PLSSVM classes exposed on the Python side:

#### `plssvm.Parameter`

The parameter class encapsulates all necessary hyper-parameters needed to fit an SVM.

| constructors                                                                                            | description                                                                      |
|---------------------------------------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| `Parameter()`                                                                                           | Default construct a parameter object.                                            |
| `Parameter(kernel_type, degree, gamma, coef0, cost)`                                                    | Construct a parameter object by explicitly providing each hyper-parameter value. |
| `Parameter([kernel_type=KernelFunctionType.LINEAR, degree=3, gamma=*1/#features*, coef=0.0, cost=1.0])` | Construct a parameter object with the provided named parameters.                 |

| attributes                         | description                                                                                  |
|------------------------------------|----------------------------------------------------------------------------------------------|
| `kernel_type : KernelFunctionType` | The used kernel function type (default: `LINEAR`).                                           |
| `degree : int`                     | The used degree in the polynomial kernel function (default: `3`).                            |
| `gamma : real_type`                | The used gamma in the polynomial and rbf kernel functions (default: `1 / #features`).        |
| `coef0 : real_type`                | The used coef0 in the polynomial kernel function (default: `0.0`).                           |
| `cost : real_type`                 | The used cost factor applied to the kernel matrix's diagonal by `1 / cost` (default: `1.0`). |

| methods               | description                                                                                         |
|-----------------------|-----------------------------------------------------------------------------------------------------|
| `equivalent(params2)` | Check whether the two parameter objects are equivalent. Same as `plssvm.equivalent(self, params2)`. |
| `param1 == param2`    | Check whether two parameter objects are identical.                                                  |
| `param1 != param2`    | Check whether two parameter objects aren't identical.                                               |
| `print(param)`        | Overload to print a `plssvm.Parameter` object displaying the used hyper-parameter.                  |

#### `plssvm.DataSet`

A class encapsulating a used data set.
The label type of `plssvm.DataSet` corresponds to the value of `-DPLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE` as provided during PLSSVM's build step (default: `std::string`).
If another label type is desired, one can simply use, e.g., `plssvm.DataSet_intc` for a data set with plain integers as label type.

| constructors                                                                                         | description                                                                                                                                                                                                                                     |
|------------------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `DataSet(filename, [file_format=*depending on the extesion of the filename*, scaling=*no scaling*])` | Construct a new data set using the data provided in the given file. Default file format: determines the file content based on its extension (.arff, everything else assumed to be a LIBSVM file). Default scaling: don't scale the data points. |
| `DataSet(data, [scaling=*no scaling*])`                                                              | Construct a new data set using the provided data directly. Default scaling: don't scale the data points.                                                                                                                                        |
| `DataSet(data, labels, [scaling=*no scaling*])`                                                      | Construct a new data set using the provided data and labels directly. Default scaling: don't scale the data points.                                                                                                                             |

| methods             | description                                                                                                                                                |
|---------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `save(filename)`    | Save the current data set to the provided file.                                                                                                            |
| `num_data_points()` | Return the number of data points in the data set.                                                                                                          |
| `num_features()`    | Return the number of features in the data set.                                                                                                             |
| `data()`            | Return the data points.                                                                                                                                    |
| `has_labels()`      | Check whether the data set is annotated with labels.                                                                                                       |
| `labels()`          | Return the labels, if present.                                                                                                                             |
| `num_classes()`     | Return the number of classes. **Note**: `0` if no labels are present.                                                                                      |
| `classes()`         | Return the different classes, if labels are present.                                                                                                       |
| `is_scaled()`       | Check whether the data points have been scaled.                                                                                                            |
| `scaling_factors()` | Return the scaling factors, if the data set has been scaled.                                                                                               |
| `print(param)`      | Overload to print a `plssvm.DataSet` object displaying the number of data points and features as well as the classes and scaling interval (if applicable). |

##### `plssvm.DataSetScaling`

A class encapsulating and performing the scaling of a `plssvm.DataSet`.
The label type of `plssvm.DataSetScaling` corresponds to the value of `-DPLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE` as provided during PLSSVM's build step (default: `std::string`).
If another label type is desired, one can simply use, e.g., `plssvm.DataSetScaling_intc` for a data set with plain integers as label  type.

| constructors                   | description                                                          |
|--------------------------------|----------------------------------------------------------------------|
| `DataSetScaling(lower, upper)` | Scale all data points feature-wise to the interval `[lower, upper]`. |
| `DataSetScaling(interval)`     | Scale all data points feature-wise to the provided interval.         |
| `DataSetScaling(filename)`     | Read previously calculated scaling factors from the provided file.   |

| attributes                                                        | description                                  |
|-------------------------------------------------------------------|----------------------------------------------|
| `scaling_interval : tuple of real_type of shape (2,)`             | The scaling interval.                        |
| `scaling_factors : numpy.ndarray of plssvm.DataSetScalingFactors` | The calculated feature-wise scaling factors. |

| methods          | description                                                                                                       |
|------------------|-------------------------------------------------------------------------------------------------------------------|
| `save(filename)` | Save the current scaling factors to the provided file.                                                            |
| `print(scaling)` | Overload to print a `plssvm.DataSetScaling` object displaying the scaling interval and number of scaling factors. |

##### `plssvm.DataSetScalingFactors`

A class encapsulating a scaling factor for a specific feature in a data set.
The label type of `plssvm.DataSetScalingFactors` corresponds to the value of `-DPLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE` as provided during PLSSVM's build step (default: `std::string`).
If another label type is desired, one can simply use, e.g., `plssvm.DataSetScalingFactors_intc` for a data set with plain integers as label type.
**Note**: it shouldn't be necessary to directly use `plssvm.DataSetScalingFactors` in user code.

| constructors                                         | description                                                                                           |
|------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| `DataSetScalingFactors(feature_index, lower, upper)` | Construct a new scaling factor for the provided feature with the features minimum and maximum values. |

| attributes            | description                                     |
|-----------------------|-------------------------------------------------|
| `feature : size_type` | The index of the current feature.               |
| `lower : real_type`   | The minimum value of the current feature index. |
| `upper : real_type`   | The maximum value of the current feature index. |

| methods                 | description                                                                                                    |
|-------------------------|----------------------------------------------------------------------------------------------------------------|
| `print(scaling_factor)` | Overload to print a `plssvm.DataSetScaling` object displaying the feature's index, minimum, and maximum value. |

#### `plssvm.CSVM`

The main class responsible for fitting an SVM model and later predicting or scoring new data sets.
It uses either the provided backend type or the default determined one to create a PLSSVM CSVM of the correct backend type.
**Note**: the backend specific CSVMs are only available if the respective backend has been enabled during PLSSVM's build step.
These backend specific CSVMs can also directly be used, e.g., `plssvm.CSVM(plssvm.BackendType.CUDA)` is equal to `plssvm.cuda.CSVM` (the same also holds for all other backends).
If the most performant backend should be used, it is sufficient to use `plssvm.CSVM()`.

| constructors                                                        | description                                                                                                                                            |
|---------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CSVM([backend, target_platform, plssvm.Parameter kwargs])`         | Create a new CSVM with the provided named arguments.                                                                                                   |
| `CSVM(params, [backend, target_platform, plssvm.Parameter kwargs])` | Create a new CSVM with the provided parameters and named arguments; the values in the `plssvm.Parameter` will be overwritten by the keyword arguments. |

**Note**: if the backend type is `plssvm.BackendType.SYCL` two additional named parameters can be provided: 
`sycl_implementation_type` to choose between DPC++ and AdaptiveCpp as SYCL implementations and `sycl_kernel_invocation_type` to choose between the two different SYCL kernel invocation types.

| methods                                                                                                                                      | description                                                                                                                                                                                                        |
|----------------------------------------------------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `set_params(params)`                                                                                                                         | Replace the current `plssvm.Parameter` with the provided one.                                                                                                                                                      |
| `set_params([kernel_type=KernelFunctionType.LINEAR, degree=3, gamma=*1/#features*, coef=0.0, cost=1.0])`                                     | Replace the current `plssvm.Parameter` values with the provided named parameters.                                                                                                                                  |
| `get_params()`                                                                                                                               | Return the `plssvm.Parameter` that are used in the CSVM to learn the model.                                                                                                                                        |
| `get_target_platform()`                                                                                                                      | Return the target platfrom this CSVM is running on.                                                                                                                                                                |
| `fit(data_set, [epsilon=0.01, classification=plssvm.ClassificatioType.OAA, solver=plssvm.SolverType.AUTOMATIC, max_iter=*#datapoints - 1*])` | Learn a LSSVM model given the provided data points and optional parameters (the termination criterion in the CG algorithm, the classification strategy, the used solver, and the maximum number of CG iterations). |
| `predict(model, data_set)`                                                                                                                   | Predict the labels of the data set using the previously learned model.                                                                                                                                             |
| `score(model)`                                                                                                                               | Score the model with respect to itself returning its accuracy.                                                                                                                                                     |
| `score(model, data_set)`                                                                                                                     | Score the model given the provided data set returning its accuracy.                                                                                                                                                |

#### `plssvm.openmp.CSVM`, `plssvm.cuda.CSVM`, `plssvm.hip.CSVM`, `plssvm.opencl.CSVM`, `plssvm.sycl.CSVM`, `plssvm.dpcpp.CSVM`, `plssvm.adaptivecpp.CSVM`

These classes represent the backend specific CSVMs.
**Note**: they are only available if the respective backend has been enabled during PLSSVM's build step.
**Note**: the `plssvm.sycl.CSVM` is equal to the respective `plssvm.dpcpp.CSVM` or `plssvm.adaptivecpp.CSVM` if only one SYCL implementation is available or the SYCL implementation defined by `-DPLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION` during PLSSVM's build step.
These classes inherit all methods from the base `plssvm.CSVM` class.

| constructors                              | description                                                                                                                                  |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------|
| `CSVM()`                                  | Create a new CSVM with the default target platform. The hyper-parameters are set to their default values.                                    |
| `CSVM([plssvm.Parameter kwargs])`         | Create a new CSVM with the default target platform. The hyper-parameter values are set ot the provided named parameter values.               |
| `CSVM(params)`                            | Create a new CSVM with the default target platform. The hyper-parameters are explicitly set to the provided `plssvm.Parameter`.              |
| `CSVM(target)`                            | Create a new CSVM with the default the provided target platform. The hyper-parameters are set to their default values.                       |
| `CSVM(target, [plssvm.Parameter kwargs])` | Create a new CSVM with the default the provided target platform. The hyper-parameter values are set ot the provided named parameter values.  |
| `CSVM(target, params)`                    | Create a new CSVM with the default the provided target platform. The hyper-parameters are explicitly set to the provided `plssvm.Parameter`. |

In case of the SYCL CSVMs (`plssvm.sycl.CSVM`, `plssvm.dpcpp.CSVM`, and `plssvm.adaptivecpp.CSVM`) the additional named argument `sycl_kernel_invocation_type` to choose between the two different SYCL kernel invocation types can be provided.

Except for the `plssvm.openmp.CSVM` the following methods are additional available for the backend specific CSVMs.

| methods                    | description                                                                                                                           |
|----------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| `num_available_devices()`  | Return the number of available devices, i.e., if the target platform represents a GPU, this function returns the number of used GPUs. |

In case of the SYCL CSVMs (`plssvm.sycl.CSVM`, `plssvm.dpcpp.CSVM`, and `plssvm.adaptivecpp.CSVM`) the following methods are additional available for the backend specific CSVMs.

| methods                        | description                             |
|--------------------------------|-----------------------------------------|
| `get_kernel_invocation_type()` | Return the SYCL kernel invocation type. |

#### `plssvm.Model`

A class encapsulating a model learned during a call to `plssvm.CSVM.fit()`.
The label type of `plssvm.Model` corresponds to the value of `-DPLSSVM_PYTHON_BINDINGS_PREFERRED_LABEL_TYPE` as provided during PLSSVM's build step (default: `std::string`).
If another label type is desired, one can simply use, e.g., `plssvm.Model_intc` for a model with plain integers as label type.

| constructors        | description                                                                     |
|---------------------|---------------------------------------------------------------------------------|
| `Model(model_file)` | Construct a new model object by reading a previously learned model from a file. |

| methods                     | description                                                                                                                                                      |
|-----------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `save(filename)`            | Save the current model to the provided file.                                                                                                                     |
| `num_support_vectors()`     | Return the number of support vectors. **Note**: for LSSVMs this corresponds to the number of training data points.                                               |
| `num_features()`            | Return the number of features each support vector has.                                                                                                           |
| `get_params()`              | Return the `plssvm.Parameter` that were used to learn this model.                                                                                                |
| `support_vectors()`         | Return the support vectors learned in this model. **Note**: for LSSVMs this corresponds to all training data points.                                             |
| `labels()`                  | Return the labels of the support vectors.                                                                                                                        |
| `num_classes()`             | Return the number of different classes.                                                                                                                          |
| `weights()`                 | Return the learned weights.                                                                                                                                      |
| `rho()`                     | Return the learned bias values.                                                                                                                                  |
| `get_classification_type()` | Return the used classification strategy.                                                                                                                         |
| `print(model)`              | Overload to print a `plssvm.Model` object displaying the number of support vectors and features, as well as the learned biases and used classification strategy. |

#### `plssvm.Version`

A class encapsulating the version information of the used PLSSVM installation.

| attributes         | description                               |
|--------------------|-------------------------------------------|
| `name : string`    | The full name of the PLSSVM library.      |
| `version : string` | The PLSSVM version ("major.minor.patch"). |
| `major : int`      | The major PLSSVM version.                 |
| `minor : int`      | The minor PLSSVM version.                 |
| `patch : int`      | The patch PLSSVM version.                 |

#### `plssvm.detail.PerformanceTracker`

A submodule used to track various performance statistics like runtimes, but also the used setup and hyper-parameters.
The tracked metrics can be saved to a YAML file for later post-processing.
**Note**: only available if PLSSVM was built with `-DPLSSVM_ENABLE_PERFORMANCE_TRACKING=ON`!

| function                                           | description                                                                      |
|----------------------------------------------------|----------------------------------------------------------------------------------|
| `add_string_tracking_entry(category, name, value)` | Add a new tracking entry to the provided category with the given name and value. |
| `add_parameter_tracking_entry(params)`             | Add a new tracking entry for the provided `plssvm.Parameter` object.             |
| `pause()`                                          | Pause the current performance tracking.                                          |
| `resume()`                                         | Resume performance tracking.                                                     |
| `save(filename)`                                   | Save all collected tracking information to the provided file.                    |

### Free functions

The following table lists all free functions in PLSSVM directly callable via `plssvm.`.

| function                                                                    | description                                                                                                                                                                                                                                                                                       |
|-----------------------------------------------------------------------------|---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `list_available_target_platforms()`                                         | List all available target platforms (determined during PLSSVM's build step).                                                                                                                                                                                                                      |
| `determine_default_target_platform(platform_device_list)`                   | Determines the default target platform used given the available target platforms.                                                                                                                                                                                                                 |
| `kernel_function_type_to_math_string(kernel)`                               | Returns a math string of the provided kernel function.                                                                                                                                                                                                                                            |
| `linear_kernel_function(x, y)`                                              | Calculate the linear kernel function of two vectors: x'*y                                                                                                                                                                                                                                         |
| `polynomial_kernel_function(x, y, degree, gamma, coef0)`                    | Calculate the polynomial kernel function of two vectors: (gamma*x'*y+coef0)^degree, with degree ∊ ℤ, gamma > 0                                                                                                                                                                                    |
| `rbf_kernel_function(x, y, gamma)`                                          | Calculate the radial basis function kernel function of two vectors: exp(-gamma*\|x-y\|^2), with gamma > 0                                                                                                                                                                                         |
| `kernel_function(x, y, params)`                                             | Calculate the kernel function provided in params with the additional parameters also provided in params.                                                                                                                                                                                          |
| `classification_type_to_full_string(classification)`                        | Returns the full string of the provided classification type, i.e., "one vs. all" and "one vs. one" instead of only "oaa" or "oao".                                                                                                                                                                |
| `calculate_number_of_classifiers(classification, num_classes)`              | Return the number of necessary classifiers in a multi-class setting with the provided classification strategy and number of different classes.                                                                                                                                                    |
| `list_available_backends()`                                                 | List all available backends (determined during PLSSVM's build step).                                                                                                                                                                                                                              |
| `determine_default_backend(available_backends, available_target_platforms)` | Determines the default backend used given the available backends and target platforms.                                                                                                                                                                                                            |
| `quiet()`                                                                   | Supress **all** command line output of PLSSVM functions.                                                                                                                                                                                                                                          | 
| `get_verbosity()`                                                           | Return the current verbosity level.                                                                                                                                                                                                                                                               |
| `set_verbosity(verbosity)`                                                  | Explicitly set the current verbosity level. `plssvm.set_verbosity(plssvm.VerbosityLevel.QUIET)` is equal to `plssvm.quiet()`.                                                                                                                                                                     |
| `equivalent(params1, params2)`                                              | Check whether the two parameter classes are equivalent, i.e., the parameters for **the current kernel function** are identical. E.g., for the rbf kernel function the gamma values must be identical, but the degree values can be different, since degree isn't used in the rbf kernel function. |

If a SYCL implementation is available, additional free functions are available:

| function                                 | description                                                                      |
|------------------------------------------|----------------------------------------------------------------------------------|
| `list_available_sycl_implementations()`  | List all available SYCL implementations (determined during PLSSVM's build step). |

### Exceptions

The PLSSVM Python3 bindings define a few new exception types:

| exception                    | description                                                                                                            |
|------------------------------|------------------------------------------------------------------------------------------------------------------------|
| `PLSSVMError`                | Base class of all other PLSSVM specific exceptions.                                                                    |
| `InvalidParameterError`      | If an invalid hyper-parameter has been provided in the `plssvm.Parameter` class.                                       |
| `FileReaderError`            | If something went wrong while reading the requested file (possibly using memory mapped IO.)                            |
| `DataSetError`               | If something related to the `plssvm.DataSet` class(es) went wrong, e.g., wrong arguments provided to the constructors. |
| `FileNotFoundError`          | If the requested data or model file couldn't be found.                                                                 |
| `InvalidFileFormatError`     | If the requested data or model file are invalid, e.g., wrong LIBSVM model header.                                      |
| `UnsupportedBackendError`    | If an unsupported backend has been requested.                                                                          |
| `UnsupportedKernelTypeError` | If an unsupported target platform has been requested.                                                                  |
| `GPUDevicePtrError`          | If something went wrong in one of the backend's GPU device pointers. **Note**: shouldn't occur in user code.           |
| `MatrixError`                | If something went wrong in the internal matrix class. **Note**: shouldn't occur in user code.                          |

Depending on the available backends, additional `BackendError`s are also available (e.g., `plssvm.cuda.BackendError`).