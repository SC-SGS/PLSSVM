/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Functions used for testing the correctness of the PLSSVM implementation.
 */

#pragma once

#include "plssvm/detail/assert.hpp"   // PLSSVM_ASSERT
#include "plssvm/detail/utility.hpp"  // plssvm::detail::always_false_v
#include "plssvm/kernel_types.hpp"    // plssvm::kernel_type

#include <vector>  // std::vector

namespace compare {
namespace detail {

/**
 * @brief Compute the value of te two vectors @p x and @p y using the linear kernel function: \f$\vec{x}^T \cdot \vec{y}\f$.
 * @tparam real_type the type of the data
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type linear_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y);
/**
 * @brief Compute the value of te two vectors @p x and @p y using the linear kernel function: \f$\vec{x}^T \cdot \vec{y}\f$.
 * @details Adds the final results in a block scheme to mimic the calculations done in the respective backends.
 * @tparam real_type the type of the data
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @param[in] num_devices the number of devices used by the currently tested backend
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type linear_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, std::size_t num_devices);
/**
 * @brief Compute the value of te two vectors @p x and @p y using the polynomial kernel function: \f$(gamma \cdot \vec{x}^T \cdot \vec{y} + coef0)^{degree}\f$.
 * @tparam real_type the type of the data
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @param[in] degree parameter in the polynomial kernel function
 * @param[in] gamma parameter in the polynomial kernel function
 * @param[in] coef0 parameter in the polynomial kernel function
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type poly_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, int degree, real_type gamma, real_type coef0);
/**
 * @brief Compute the value of te two vectors @p x and @p y using the radial basis function kernel function: \f$e^{(-gamma \cdot |\vec{x} - \vec{y}|^2)}\f$.
 * @tparam real_type the type of the data
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @param[in] gamma parameter in the kernel function
 * @return the result after applying the rbf kernel function (`[[nodiscard]]`)
 */
template <typename real_type>
[[nodiscard]] real_type radial_kernel(const std::vector<real_type> &x, const std::vector<real_type> &y, real_type gamma);

}  // namespace detail

/**
 * @brief Computes the value of the two vectors @p x and @p y using the kernel function determined at compile-time.
 * @tparam kernel the type of the kernel
 * @tparam real_type the type of the data
 * @tparam SVM the used SVM implementation
 * @param[in] x the first vector
 * @param[in] y the second vector
 * @param[in] csvm the SVM encapsulating all necessary parameters
 * @return the result after applying the kernel function (`[[nodiscard]]`)
 */
template <plssvm::kernel_type kernel, typename real_type, typename SVM>
[[nodiscard]] real_type kernel_function(const std::vector<real_type> &x, const std::vector<real_type> &y, [[maybe_unused]] const SVM &csvm) {
    PLSSVM_ASSERT(x.size() == y.size(), "Sizes mismatch!: {} != {}", x.size(), y.size());

    if constexpr (kernel == plssvm::kernel_type::linear) {
        return detail::linear_kernel(x, y);
    } else if constexpr (kernel == plssvm::kernel_type::polynomial) {
        return detail::poly_kernel(x, y, csvm.get_degree(), csvm.get_gamma(), csvm.get_coef0());
    } else if constexpr (kernel == plssvm::kernel_type::rbf) {
        return detail::radial_kernel(x, y, csvm.get_gamma());
    } else {
        static_assert(plssvm::detail::always_false_v<real_type>, "Unknown kernel type!");
    }
}

/**
 * @brief Computes the `q` vector, a subvector of the least-squares matrix equation, using the kernel function determined at compile-time.
 * @tparam kernel the type of the kernel
 * @tparam real_type the type of the data
 * @tparam SVM the used SVM implementation
 * @param[in] data the data points
 * @param[in] num_devices the number of devices used in the currently tested backend, only interesting for the linear kernel
 * @param[in] csvm the SVM encapsulating all necessary parameters
 * @return the generated `q` vector (`[[nodiscard]]`)
 */
template <plssvm::kernel_type kernel, typename real_type, typename SVM>
[[nodiscard]] std::vector<real_type> generate_q(const std::vector<std::vector<real_type>> &data, [[maybe_unused]] const std::size_t num_devices, const SVM &csvm) {
    std::vector<real_type> result;
    result.reserve(data.size() - 1);

    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < data.size() - 1; ++i) {
        if constexpr (kernel == plssvm::kernel_type::linear) {
            result.emplace_back(detail::linear_kernel(data.back(), data[i], num_devices));
        } else {
            result.emplace_back(kernel_function<kernel>(data.back(), data[i], csvm));
        }
    }
    return result;
}

/**
 * @brief Computes the device kernel, using the kernel function determined at compile-time.
 * @tparam kernel the type of the kernel
 * @tparam real_type the type of the data
 * @tparam SVM the used SVM implementation
 * @param[in] data the data points
 * @param[in,out] x the right-hand side
 * @param[in] q the `q` vector
 * @param[in] QA_cost the bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] csvm the SVM encapsulating all necessary parameters
 * @return the result vector (`[[nodiscard]]`)
 */
template <plssvm::kernel_type kernel, typename real_type, typename SVM>
[[nodiscard]] std::vector<real_type> device_kernel_function(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &x, const std::vector<real_type> &q, const real_type QA_cost, const real_type cost, const real_type add, const SVM &csvm) {
    PLSSVM_ASSERT(x.size() == q.size(), "Sizes mismatch!: {} != {}", x.size(), q.size());
    PLSSVM_ASSERT(x.size() == data.size() - 1, "Sizes mismatch!: {} != {}", x.size(), data.size() - 1);

    using size_type = typename std::vector<real_type>::size_type;

    const size_type dept = x.size();

    std::vector<real_type> r(dept, 0.0);
    for (size_type i = 0; i < dept; ++i) {
        for (size_type j = 0; j < dept; ++j) {
            if (i >= j) {
                const real_type temp = kernel_function<kernel>(data[i], data[j], csvm) + QA_cost - q[i] - q[j];
                if (i == j) {
                    r[i] += (temp + real_type{ 1.0 } / cost) * x[i] * add;
                } else {
                    r[i] += temp * x[j] * add;
                    r[j] += temp * x[i] * add;
                }
            }
        }
    }
    return r;
}

}  // namespace compare
