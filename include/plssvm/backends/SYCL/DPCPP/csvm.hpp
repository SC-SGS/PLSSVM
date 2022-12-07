/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the SYCL backend with DPC++ as SYCL implementation.
 */

#ifndef PLSSVM_BACKENDS_SYCL_DPCPP_CSVM_HPP_
#define PLSSVM_BACKENDS_SYCL_DPCPP_CSVM_HPP_

#include "plssvm/backends/SYCL/csvm.hpp"

#include <string>       // std::string
#include <type_traits>  // std::true_type
#include <utility>      // std::forward

namespace plssvm {

namespace dpcpp {

class csvm : public ::plssvm::sycl::detail::csvm {
    using base_type = ::plssvm::sycl::detail::csvm;

  public:
    template <typename... Args>
    explicit csvm(Args &&...args) :
        base_type{ std::forward<Args>(args)... } {}

  protected:
    /**
     * @copydoc plssvm::sycl::detail::csvm::compiler_info
     */
    [[nodiscard]] std::string compiler_info() const final {
        return "DPC++";
        //        return fmt::format("DPC++; {}", __SYCL_COMPILER_VERSION);
    }
};

}  // namespace dpcpp

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs using the SYCL backend with DPC++ as SYCL implementation are available.
 */
template <>
struct csvm_backend_exists<dpcpp::csvm> : std::true_type {};

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_SYCL_DPCPP_CSVM_HPP_
