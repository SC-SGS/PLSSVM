/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements tags classes used for sklearn's `__sklearn_tags__` attribute.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_SKLEARN_LIKE_TAGS_HPP_
#define PLSSVM_BINDINGS_PYTHON_SKLEARN_LIKE_TAGS_HPP_
#pragma once

#include <optional>  // std::optional
#include <string>    // std::string
#include <vector>    // std::vector

/**
 * @brief Tags for the target data.
 */
struct TargetTags {
    /// Whether the estimator requires y to be passed to `fit`, `fit_predict`, or `fit_transform` methods.
    bool required = true;
    /// Whether the input is a 1D labels (y).
    bool one_d_labels = false;
    /// Whether the input is a 2D labels (y).
    bool two_d_labels = false;
    /// Whether the estimator requires a positive y (only applicable for regression).
    bool positive_only = false;
    /// Whether a regressor supports multi-target outputs or a classifier supports multi-class multi-output.
    bool multi_output = false;
    /// Whether the target can be single-output. This can be `false` if the estimator supports only multi-output cases.
    bool single_output = true;
};

/**
 * @brief Tags for the transformer.
 */
struct TransformerTags {
    /// Applies only on transformers.
    /// It corresponds to the data types which will be preserved such that `X_trans.dtype` is the same as `X.dtype` after calling `transformer.transform(X)`.
    std::vector<std::string> preserves_dtype{ "float64" };
};

/**
 * @brief Tags for the classifier.
 */
struct ClassifierTags {
    /// Whether the estimator fails to provide a “reasonable” test-set score, which currently for classification is an
    /// accuracy of 0.83 on `make_blobs(n_samples=300, random_state=0)`.
    bool poor_score = false;
    /// Whether the classifier can handle multi-class classification.
    bool multi_class = true;
    /// Whether the classifier supports multi-label output: a data point can be predicted to belong to a variable number of classes.
    bool multi_label = false;
};

/**
 * @brief Tags for the regressor.
 */
struct RegressorTags {
    /// Whether the estimator fails to provide a “reasonable” test-set score, which currently for regression is an R2 of 0.5 on
    /// `make_regression(n_samples=200, n_features=10, n_informative=1, bias=5.0, noise=20, random_state=42)`.
    bool poor_score = false;
};

/**
 * @brief Tags for the input data.
 */
struct InputTags {
    /// Whether the input can be a 1D array.
    bool one_d_array = false;
    /// Whether the input can be a 2D array.
    bool two_d_array = true;
    /// Whether the input can be a 3D array.
    bool three_d_array = false;
    /// Whether the input can be a sparse matrix.
    bool sparse = false;
    /// Whether the input can be categorical.
    bool categorical = false;
    /// Whether the input can be an array-like of strings.
    bool string = false;
    /// Whether the input can be a dictionary.
    bool dict = false;
    /// Whether the estimator requires positive X.
    bool positive_only = false;
    /// Whether the estimator supports data with missing values encoded as `np.nan`.
    bool allow_nan = false;
    /// This boolean attribute indicates whether the data(`X`), fit and similar methods consists of pairwise measures
    /// over samples rather than a feature representation for each sample. It is usually `true` where an estimator has
    /// a metric or affinity or kernel parameter with value "precomputed".
    bool pairwise = false;
};

/**
 * @brief Tags for the estimator.
 */
struct Tags {
    /// The type of the estimator. Can be one of: - “classifier” - “regressor” - “transformer” - “clusterer” - “outlier_detector” - “density_estimator”
    std::optional<std::string> estimator_type;
    /// The target(y) tags.
    TargetTags target_tags;
    /// The transformer tags.
    std::optional<TransformerTags> transformer_tags;
    /// The classifier tags.
    std::optional<ClassifierTags> classifier_tags;
    /// The regressor tags.
    std::optional<RegressorTags> regressor_tags;
    /// Whether the estimator supports Array API compatible inputs.
    bool array_api_support = false;
    /// Whether the estimator skips input-validation. This is only meant for stateless and dummy transformers!
    bool no_validation = false;
    /// Whether the estimator is not deterministic given a fixed `random_state`.
    bool non_deterministic = false;
    /// Whether the estimator requires to be fitted before calling one of `transform`, `predict`, `predict_proba`, or `decision_function`.
    bool requires_fit = true;
    /// Whether to skip common tests entirely. Don’t use this unless you have a very good reason.
    bool _skip_test = false;
    /// The input data(X) tags.
    InputTags input_tags;
};

#endif  // PLSSVM_BINDINGS_PYTHON_SKLEARN_LIKE_TAGS_HPP_
