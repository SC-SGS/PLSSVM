/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the SYCL backend with hipSYCL as SYCL implementation.
 */

#ifndef PLSSVM_BACKENDS_SYCL_HIPSYCL_CSVM_HPP_
#define PLSSVM_BACKENDS_SYCL_HIPSYCL_CSVM_HPP_

#include "plssvm/backends/SYCL/csvm.hpp"

#include <string>       // std::string
#include <type_traits>  // std::true_type
#include <utility>      // std::forward

#include <iostream>

namespace plssvm {

namespace hipsycl {

class csvm : public ::plssvm::sycl::detail::csvm {
    using base_type = ::plssvm::sycl::detail::csvm;

  public:
    template <typename... Args>
    explicit csvm(Args &&...args) :
        base_type{ std::forward<Args>(args)... } {}

    static constexpr bool is_preferred() {
#if PLSSVM_SYCL_BACKEND_PREFERRED_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL
        return true;
#else
        return false;
#endif
    }

  protected:
    /**
     * @copydoc plssvm::sycl::detail::csvm::compiler_info
     */
    [[nodiscard]] std::string compiler_info() const final {
        return "hipSYCL";
        //        return fmt::format("hipSYCL; {}", ::hipsycl::sycl::detail::version_string());
    }
};

}  // namespace hipsycl

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs using the SYCL backend with hipSYCL as SYCL implementation are available.
 */
template <>
struct csvm_backend_exists<hipsycl::csvm> : std::true_type {};

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_SYCL_HIPSYCL_CSVM_HPP_
