/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief MOCK class for the C-SVM class using the SYCL backend.
 */

#pragma once

#include "plssvm/backends/SYCL/csvm.hpp"               // plssvm::sycl::csvm
#include "plssvm/backends/SYCL/detail/device_ptr.hpp"  // plssvm::sycl::detail::device_ptr
#include "plssvm/kernel_types.hpp"                     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/target_platform.hpp"                  // plssvm::target_platform

#include "sycl/sycl.hpp"  // SYCL stuff

#include <vector>  // std::vector

/**
 * @brief GTest mock class for the SYCL C-SVM.
 * @tparam T the type of the data
 */
template <typename T>
class mock_sycl_csvm : public plssvm::sycl::csvm<T> {
    using base_type = plssvm::sycl::csvm<T>;

  public:
    using real_type = typename base_type::real_type;
    using size_type = typename base_type::size_type;

    explicit mock_sycl_csvm(const plssvm::parameter<T> &params) :
        base_type{ params } {}
    explicit mock_sycl_csvm(const plssvm::target_platform target, const plssvm::kernel_type kernel, const real_type degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
        base_type{ target, kernel, degree, gamma, coef0, cost, epsilon, print_info } {}

    // make non-virtual functions publicly visible
    using base_type::generate_q;
    using base_type::learn;
    using base_type::run_device_kernel;
    using base_type::setup_data_on_device;

    // parameter setter
    void set_cost(const real_type cost) { cost_ = cost; }
    void set_QA_cost(const real_type QA_cost) { QA_cost_ = QA_cost; }

    // getter for internal variables
    const std::vector<plssvm::sycl::detail::device_ptr<real_type>> &get_device_data() const { return data_d_; }
    std::vector<sycl::queue> &get_devices() { return devices_; }

  private:
    using base_type::cost_;
    using base_type::QA_cost_;

    using base_type::data_d_;
    using base_type::devices_;
};