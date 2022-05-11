/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends and implements the functionality shared by all of them.
 */

#pragma once

#include "plssvm/kernel_types.hpp"      // plssvm::kernel_type
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform
#include "parameter.hpp"                // plssvm::parameter

#include <cstddef>      // std::size_t
#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

#include "NamedType/named_type.hpp"
#include <iostream>
#include <memory>
#include <optional>
#include <functional>
#include "plssvm/backend_types.hpp"                         // plssvm::backend_type
#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl_generic::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl_generic::kernel_invocation_type
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

namespace plssvm {

// forward declare class TODO
//template <typename T>
//class parameter;

/**
 * @brief Base class for all C-SVM backends.
 * @tparam T the type of the data
 */
template <typename T>
class csvm {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;

    //*************************************************************************************************************************************//
    //                                                      special member functions                                                       //
    //*************************************************************************************************************************************//
    /**
     * @brief Construct a new C-SVM with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::exception if the given data pointer is the `nullptr`
     * @throws plssvm::exception if the data matrix is empty
     * @throws plssvm::exception if not all points in the data matrix have the same number of features
     * @throws plssvm::exception if no features are provided for the data points
     * @throws plssvm::exception if weights are given, but the number of weights doesn't match the number of data points
     */
    explicit csvm(const parameter<T> &params);

    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~csvm() = default;

    /**
     * @brief Delete expensive copy-constructor to make csvm a move only type.
     */
    csvm(const csvm &) = delete;
    /**
     * @brief Move-constructor as csvm is a move-only type.
     */
    csvm(csvm &&) noexcept = default;
    /**
     * @brief Delete expensive copy-assignment-operator to make csvm a move only type.
     */
    csvm &operator=(const csvm &) = delete;
    /**
     * @brief Move-assignment-operator as csvm is a move-only type.
     * @return `*this`
     */
    csvm &operator=(csvm &&) noexcept = default;

    //*************************************************************************************************************************************//
    //                                                             IO functions                                                            //
    //*************************************************************************************************************************************//

    /**
     * @brief Write the calculated model to the file denoted by @þ filename.
     * @details Writes the model using the LIBSVM format:
     * @code
     * svm_type c_svc
     * kernel_type linear
     * nr_class 2
     * total_sv 5
     * rho 0.37332362
     * label 1 -1
     * nr_sv 2 3
     * SV
     * -0.17609704 0:-1.117828e+00 1:-2.908719e+00 2:6.663834e-01 3:1.097883e+00
     * 0.883819 0:-5.282118e-01 1:-3.358810e-01 2:5.168729e-01 3:5.460446e-01
     * -0.47971326 0:-2.098121e-01 1:6.027694e-01 2:-1.308685e-01 3:1.080525e-01
     * -0.23146635 0:5.765022e-01 1:1.014056e+00 2:1.300943e-01 3:7.261914e-01
     * 0.0034576654 0:1.884940e+00 1:1.005186e+00 2:2.984999e-01 3:1.646463e+00
     * @endcode
     * @param[in] filename name of the file to write the model information to
     * @throws plssvm::exception if a call to learn() is missing
     * @throws plssvm::exception if no labels are given
     * @throws plssvm::exception if the number of labels and number of data points mismatch
     */
    void write_model(const std::string &filename);

    //*************************************************************************************************************************************//
    //                                                             learn model                                                             //
    //*************************************************************************************************************************************//
    /**
     * @brief Learns the support vectors given the data in the provided parameter class.
     * @details Performs 2 steps:
     * 1. Load the data onto the used device (e.g. one or more GPUs)
     * 2. Learn the model by solving a minimization problem using the Conjugated Gradients algorithm
     *
     * @throws plssvm::exception if no labels are given for training
     * @throws plssvm::exception if the number of labels and number of data points mismatch
     */
    void learn();

    //*************************************************************************************************************************************//
    //                                                               predict                                                               //
    //*************************************************************************************************************************************//

    /**
     * @brief Evaluates the model on the data used for training.
     * @throws plssvm::exception if no labels are given for the accuracy calculation
     * @throws plssvm::exception if no weights are provided for calculating the accuracy (possibly a call to learn() is missing)
     * @return the fraction of correctly labeled training data in percent (`[[nodiscard]]`)
     */
    [[nodiscard]] real_type accuracy();
    /**
     * @brief Evaluate the model on the given data @p point with @p correct_label being the correct label.
     * @param[in] point the data point to predict
     * @param[in] correct_label the correct label
     * @throws plssvm::exception if the number of features in @p point doesn't match the number of features in the data matrix
     * @throws plssvm::exception if no weights are provided for calculating the accuracy (possibly a call to learn() is missing)
     * @return `1.0` if @p point is predicted correctly, `0.0` otherwise. (`[[nodiscard]]`)
     */
    [[nodiscard]] real_type accuracy(const std::vector<real_type> &point, real_type correct_label);
    /**
     * @brief Evaluate the model on the given data @p points with @p correct_labels being the correct labels.
     * @param[in] points the data points to predict
     * @param[in] correct_labels the correct labels
     * @throws plssvm::exception if the number of points to predict mismatch the number of provided, correct label
     * @throws plssvm::exception if not all @p points to predict have the same number of features
     * @throws plssvm::exception if the number of features per point to predict and per point in data matrix mismatch
     * @throws plssvm::exception if no weights are provided for calculating the accuracy (possibly a call to learn() is missing)
     * @return the fraction of correctly labeled data points. (`[[nodiscard]]`)
     */
    [[nodiscard]] real_type accuracy(const std::vector<std::vector<real_type>> &points, const std::vector<real_type> &correct_labels);

    /**
     * @brief Uses the already learned model to predict a (new) data point.
     * @param[in] point the data point to predict
     * @throws plssvm::exception if the number of features in @p point doesn't match the number of features in the data matrix
     * @throws plssvm::exception if no weights are provided for calculating the accuracy (possibly a call to learn() is missing)
     * @return a negative #real_type value if the prediction for data point point is the negative class and a positive #real_type value otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] real_type predict(const std::vector<real_type> &point);

    /**
     * @brief Uses the already learned model to predict the class of a (new) data point.
     * @param[in] point the data point to predict
     * @throws plssvm::exception if the number of features in @p point doesn't match the number of features in the data matrix
     * @throws plssvm::exception if no weights are provided for calculating the accuracy (possibly a call to learn() is missing)
     * @return -1.0 if the prediction for @p point is the negative class and +1 otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] real_type predict_label(const std::vector<real_type> &point);
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] points the data points to predict
     * @throws plssvm::exception if not all @p points to predict have the same number of features
     * @throws plssvm::exception if the number of features per point to predict and per point in data matrix mismatch
     * @throws plssvm::exception if no weights are provided for calculating the accuracy (possibly a call to learn() is missing)
     * @return a [`std::vector<real_type>`](https://en.cppreference.com/w/cpp/container/vector) filled with -1 for each prediction for a data point with the negative class and +1 otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<real_type> predict_label(const std::vector<std::vector<real_type>> &points);

  protected:
    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    /**
     * @brief Initialize the data on the respective device(s) (e.g. GPUs).
     */
    virtual void setup_data_on_device() = 0;
    /**
     * @brief Generate the vector `q`, a subvector of the least-squares matrix equation.
     * @return the generated `q` vector (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<real_type> generate_q() = 0;
    /**
     * @brief Solves the equation \f$Ax = b\f$ using the Conjugated Gradients algorithm.
     * @details Solves using a slightly modified version of the CG algorithm described by [Jonathan Richard Shewchuk](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf):
     * \image html cg.png
     * @param[in] b the right-hand side of the equation \f$Ax = b\f$
     * @param[in] imax the maximum number of CG iterations
     * @param[in] eps error tolerance
     * @param[in] q subvector of the least-squares matrix equation
     * @return the alpha values
     */
    virtual std::vector<real_type> solver_CG(const std::vector<real_type> &b, std::size_t imax, real_type eps, const std::vector<real_type> &q) = 0;
    /**
     * @brief Updates the normal vector #w_, used to speed-up the prediction in case of the linear kernel function, to the current data and alpha values.
     */
    virtual void update_w() = 0;
    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] points the data points to predict
     * @return a [`std::vector<real_type>`](https://en.cppreference.com/w/cpp/container/vector) filled with negative values for each prediction for a data point with the negative class and positive values otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::vector<real_type> predict(const std::vector<std::vector<real_type>> &points) = 0;

    //*************************************************************************************************************************************//
    //                                                          kernel functions                                                           //
    //*************************************************************************************************************************************//
    /**
     * @brief Computes the value of the two vectors @p xi and @p xj using the kernel function specified during construction.
     * @param[in] xi the first vector
     * @param[in] xj the second vector
     * @throws plssvm::unsupported_kernel_type_exception if the kernel type cannot be recognized
     * @return the value computed by the kernel function (`[[nodiscard]]`)
     */
    [[nodiscard]] real_type kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj);

    /**
     * @brief Transforms @p num_points entries of the 2D data from AoS to a 1D SoA layout and adding @p boundary points.
     * @param[in] matrix the 2D vector to be transformed into a 1D representation
     * @param[in] boundary the number of boundary points
     * @param[in] num_points the number of data points of the 2D vector to transform
     * @attention Boundary values can contain random numbers!
     * @return an 1D vector in a SoA layout (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<real_type> transform_data(const std::vector<std::vector<real_type>> &matrix, std::size_t boundary, std::size_t num_points);

    //*************************************************************************************************************************************//
    //                                              parameter initialized by the constructor                                               //
    //*************************************************************************************************************************************//
    /// The target platform.
    const target_platform target_;
    /// The used kernel function.
    const kernel_type kernel_;
    /// The degree parameter used in the polynomial kernel function.
    const int degree_;
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    real_type gamma_;
    /// The coef0 parameter used in the polynomial kernel function.
    const real_type coef0_;
    /// The cost parameter in the C-SVM.
    real_type cost_;
    /// The error tolerance parameter for the CG algorithm.
    const real_type epsilon_;
    /// If `true` additional information (e.g. timing information) will be printed during execution.
    const bool print_info_;

    /// The data used the train the SVM.
    const std::shared_ptr<const std::vector<std::vector<real_type>>> data_ptr_{};
    /// The labels associated to each data point.
    std::shared_ptr<const std::vector<real_type>> value_ptr_{};
    /// The result of the CG calculation: the weights of the support vectors.
    std::shared_ptr<const std::vector<real_type>> alpha_ptr_{};

    //*************************************************************************************************************************************//
    //                                                         internal variables                                                          //
    //*************************************************************************************************************************************//
    /// The number of data points in the data set.
    std::size_t num_data_points_{};
    /// The number of features per data point.
    std::size_t num_features_{};
    /// The bias after learning.
    real_type bias_{};
    /// The bottom right matrix entry multiplied by cost.
    real_type QA_cost_{};
    /// The normal vector used for speeding up the prediction in case of the linear kernel function.
    std::vector<real_type> w_{};
};

extern template class csvm<float>;
extern template class csvm<double>;

namespace new_ {

inline bool print_info = true;

enum class file_format_type {
    libsvm,
    arff
};

template <typename T>
class parameter {
  public:
    using real_type = T;

    parameter() = default;

    kernel_type kernel = kernel_type::linear;
    int degree = 3;
    real_type gamma = real_type{ 0.0 };
    real_type coef0 = real_type{ 0.0 };
    real_type cost = real_type{ 1.0 };
    real_type epsilon = static_cast<real_type>(0.001);
    backend_type backend = backend_type::automatic;
    target_platform target = target_platform::automatic;

    ::plssvm::sycl_generic::kernel_invocation_type sycl_kernel_invocation_type = ::plssvm::sycl_generic::kernel_invocation_type::automatic;
    ::plssvm::sycl_generic::implementation_type sycl_implementation_type = ::plssvm::sycl_generic::implementation_type::automatic;
};

// FINISHED
template <typename T>
class data_set {
    using data_matrix_type = std::vector<std::vector<T>>;
    using label_vector_type = std::vector<int>;

  public:
    using real_type = T;
    using label_type = int;
    using size_type = std::size_t;

    explicit data_set(const std::string& filename);

    explicit data_set(data_matrix_type &&X) : X_ptr_{ std::make_shared<data_matrix_type>(std::move(X)) } {
        if (X_ptr_->empty()) {
            throw std::runtime_error("empty matrix"); // TODO: correct exception
        }
    }
    data_set(data_matrix_type &&X, label_vector_type &&y) : X_ptr_{ std::make_shared<data_matrix_type>(std::move(X)) }, y_ptr_{ std::make_shared<label_vector_type>(std::move(y)) } {
        // TODO: exception?? if size != size
    }
    // save the data set in the given format
    void save_data_set(const std::string& filename, file_format_type format);

    // scale data features to be in range [lower, upper]
    void scale(real_type lower, real_type upper);

    [[nodiscard]] const data_matrix_type& data() const noexcept { return *X_ptr_; }
    [[nodiscard]] std::optional<std::reference_wrapper<const label_type>> labels() const noexcept {
        return this->has_labels() ? *y_ptr_ : std::nullopt;
    }
    [[nodiscard]] bool has_labels() const noexcept { return y_ptr_ != nullptr; }

    [[nodiscard]] size_type num_data_points() const noexcept { return X_ptr_->size(); }
    [[nodiscard]] size_type num_features() const noexcept { return X_ptr_->front().size(); }

  private:
    std::shared_ptr<data_matrix_type> X_ptr_{ nullptr };
    std::shared_ptr<label_vector_type> y_ptr_{ nullptr };
};

template <typename T>
class csvm_model {
    using alpha_vector_type = std::vector<T>;

  public:
    using real_type = T;

    // read model from file
    explicit csvm_model(const std::string& filename);

    // save model to file
    void save_model(const std::string& filename);

    // predict labels of the data_set
    std::vector<typename data_set<real_type>::label_type> predict(const data_set<real_type> &data);
    // predict LS-SVM values of the data_set
    std::vector<real_type> predict_values(const data_set<real_type> &data);

    // calculate the accuracy of the model
    real_type score();
    // calculate the accuracy of the data_set
    real_type score(const data_set<real_type> &data);

  private:
    csvm_model(parameter<real_type> params, data_set<real_type> data) : params_{ std::move(params) }, data_{ std::move(data) }, alphas_{ std::make_shared<alpha_vector_type>() } {}

    parameter<real_type> params_;
    data_set<real_type> data_;

    std::shared_ptr<alpha_vector_type> alphas_;
    real_type rho_{ 0.0 };
};

template <typename T>
class csvm {
    friend csvm_model<T>;
  public:
    using real_type = T;

    // create new SVM with the given parameters
    explicit csvm(parameter<real_type> params) : params_{ std::move(params) } {}
    csvm(real_type cost, target_platform target) : params_{} {
        params_.kernel = kernel_type::linear;
        params_.cost = cost;
        params_.backend = backend_type::openmp;
        params_.target = target;
    }
    csvm(int degree, real_type gamma, real_type coef0, real_type cost, target_platform target) : params_{} {
        params_.kernel = kernel_type::polynomial;
        params_.degree = degree;
        params_.gamma = gamma;
        params_.coef0 = coef0;
        params_.cost = cost;
        params_.backend = backend_type::openmp;
        params_.target = target;
    }
    csvm(real_type gamma, real_type cost, target_platform target) : params_{} {
        params_.kernel = kernel_type::rbf;
        params_.gamma = gamma;
        params_.cost = cost;
        params_.backend = backend_type::openmp;
        params_.target = target;
    }

    // learn model with default eps and iter values
    csvm_model<real_type> fit(const data_set<real_type> &data) {
        // TODO: implement
        params_.epsilon = parameter<real_type>{}.epsilon;
        return csvm_model<real_type>(params_, data);
    }
    // learn model until eps is reached
    csvm_model<real_type> fit(const data_set<real_type> &data, real_type eps) {
        // TODO: implement
        params_.epsilon = eps;
        return csvm_model<real_type>(params_, data);
    }
    // learn model using exact iter CG iterations
    csvm_model<real_type> fit(const data_set<real_type> &data, std::size_t iter) {
        // TODO: implement
        params_.epsilon = parameter<real_type>{}.epsilon;
        return csvm_model<real_type>(params_, data);
    }
    // learn model until eps is reached OR iter CG iterations are reached
    csvm_model<real_type> fit(const data_set<real_type> &data, real_type eps, std::size_t iter) {
        // TODO: implement
        params_.epsilon = eps;
        return csvm_model<real_type>(params_, data);
    }
    // learn model until accuracy for test data set is reached
    csvm_model<real_type> fit(const data_set<real_type> &data_train, const data_set<real_type> &data_test, real_type accuracy) {
        // TODO: implement
        params_.epsilon = parameter<real_type>{}.epsilon;
        return csvm_model<real_type>(params_, data_train);
    }

  private:
    parameter<real_type> params_;
};

}

}  // namespace plssvm
