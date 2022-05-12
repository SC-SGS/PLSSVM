/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements the parameter base class encapsulating all necessary parameters.
 */

#pragma once

#include "plssvm/backend_types.hpp"                         // plssvm::backend_type
#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl_generic::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl_generic::kernel_invocation_type
#include "plssvm/detail/arithmetic_type_name.hpp"           // plssvm::detail::arithmetic_type_name
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include <iosfwd>       // forward declare std::ostream
#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <variant>      // std::variant
#include <vector>       // std::vector

namespace plssvm {

namespace sycl {
using namespace ::plssvm::sycl_generic;
}

template <typename T>
struct parameter;


using parameter_variants = std::variant<parameter<float>, parameter<double>>;


/**
 * @brief Base class for encapsulating all necessary parameters possibly provided through command line arguments.
 * @tparam T the type of the data
 */
template <typename T>
struct parameter {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;

    /// The used kernel function: linear, polynomial or radial basis functions (rbf).
    kernel_type kernel = kernel_type::linear;
    /// The degree parameter used in the polynomial kernel function.
    int degree = 3;
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    real_type gamma = real_type{ 0.0 };
    /// The coef0 parameter used in the polynomial kernel function.
    real_type coef0 = real_type{ 0.0 };
    /// The cost parameter in the C-SVM.
    real_type cost = real_type{ 1.0 };
    /// The error tolerance parameter for the CG algorithm.
    real_type epsilon = static_cast<real_type>(0.001);
    /// The used backend: automatic (depending on the specified target_platforms), OpenMP, OpenCL, CUDA, or SYCL.
    backend_type backend = backend_type::automatic;
    /// The target platform: automatic (depending on the used backend), CPUs or GPUs from NVIDIA, AMD or Intel.
    target_platform target = target_platform::automatic;

    /// The kernel invocation type when using SYCL as backend.
    sycl::kernel_invocation_type sycl_kernel_invocation_type = sycl::kernel_invocation_type::automatic;
    /// The SYCL implementation to use with --backend=sycl.
    sycl::implementation_type sycl_implementation_type = sycl::implementation_type::automatic;

    // TODO: here?!?
    /// use strings as label type?
    bool strings_as_labels{ false };
    bool float_as_real_type{ false };
};

/**
 * @brief Output all parameters encapsulated by @p params to the given output-stream @p out.
 * @tparam T the type of the data
 * @param[in,out] out the output-stream to write the parameters to
 * @param[in] params the parameters
 * @return the output-stream
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params) {
    return out << fmt::format(
               "kernel_type                 {}\n"
               "degree                      {}\n"
               "gamma                       {}\n"
               "coef0                       {}\n"
               "cost                        {}\n"
               "epsilon                     {}\n"
               "backend                     {}\n"
               "target platform             {}\n"
               "SYCL kernel invocation type {}\n"
               "SYCL implementation type    {}\n"
               "use strings as labels       {}\n"
               "real_type                   {}\n",
               params.kernel,
               params.degree,
               params.gamma,
               params.coef0,
               params.cost,
               params.epsilon,
               params.backend,
               params.target,
               params.sycl_kernel_invocation_type,
               params.sycl_implementation_type,
               params.strings_as_labels,
               detail::arithmetic_type_name<typename parameter<T>::real_type>());
}

}  // namespace plssvm