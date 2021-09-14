/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements the base class encapsulating all necessary parameters.
 */

#pragma once

#include "plssvm/backend_types.hpp"    // plssvm::backend_type
#include "plssvm/kernel_types.hpp"     // plssvm::kernel_type
#include "plssvm/target_platform.hpp"  // plssvm::target_platform

#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace plssvm {

/**
 * @brief Class encapsulating all necessary parameters possibly provided through command line arguments.
 * @tparam T the type of the data
 */
template <typename T>
class parameter {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;
    /// Unsigned integer type.
    using size_type = std::size_t;

    /// Pure virtual, default destructor.
    virtual ~parameter() = 0;

    /// The used kernel function: linear, polynomial or radial basis functions (rbf).
    kernel_type kernel = kernel_type::linear;
    /// The degree parameter used in the polynomial kernel function.
    int degree = 3;
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    real_type gamma = 0.0;
    /// The coef0 parameter used in the polynomial kernel function.
    real_type coef0 = 0.0;
    /// The cost parameter in the C-SVM.
    real_type cost = 1.0;
    /// The error tolerance parameter for the CG algorithm.
    real_type epsilon = 0.001;
    /// If `true` additional information (e.g. timing information) will be printed during execution.
    bool print_info = true;
    /// The used backend: OpenMP, OpenCL, CUDA or SYCL.
    backend_type backend = backend_type::openmp;
    /// The target platform: automatic (depending on the used backend), CPUs or GPUs from NVIDIA, AMD or Intel.
    target_platform target = target_platform::automatic;

    /// The name of the data file to parse.
    std::string input_filename;
    /// The name of the model file to write the learned Support Vectors to.
    std::string model_filename;
    /// The name of the file to write the prediction to.
    std::string predict_filename;
};

template <typename T>
parameter<T>::~parameter() = default;

extern template class parameter<float>;
extern template class parameter<double>;

/**
 * @brief Stream-insertion operator overload for convenient printing of all parameters encapsulated by @p params.
 * @tparam T the type of the data
 * @param[in,out] out the output-stream to write the kernel type to
 * @param[in] params the parameters
 * @return the output-stream
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params);

}  // namespace plssvm