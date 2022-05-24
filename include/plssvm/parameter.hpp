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

#include "igor/igor.hpp"

#include <iosfwd>       // forward declare std::ostream
#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <variant>      // std::variant
#include <vector>       // std::vector

namespace plssvm {

// TODO: move?
namespace sycl {
using namespace ::plssvm::sycl_generic;
}

inline constexpr auto gamma = igor::named_argument<struct gamma_tag>{};
inline constexpr auto degree = igor::named_argument<struct degree_tag>{};
inline constexpr auto coef0 = igor::named_argument<struct coef0_tag>{};
inline constexpr auto cost = igor::named_argument<struct cost_tag>{};
inline constexpr auto epsilon = igor::named_argument<struct epsilon_tag>{};
inline constexpr auto max_iter = igor::named_argument<struct max_iter_tag>{};


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

    template <typename U>
    explicit operator parameter<U>() {
        // only float and doubles are allowed
        static_assert(std::is_same_v<U, float> || std::is_same_v<U, double>, "The template type can only be 'float' or 'double'!");
        return parameter<U>{ kernel, degree, static_cast<U>(gamma), static_cast<U>(coef0), static_cast<U>(cost) };
    }
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
               "real_type                   {}\n",
               params.kernel,
               params.degree,
               params.gamma,
               params.coef0,
               params.cost,
               detail::arithmetic_type_name<typename parameter<T>::real_type>());
}

}  // namespace plssvm