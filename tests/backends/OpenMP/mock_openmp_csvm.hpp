/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief MOCK class for the C-SVM class using the OpenMP backend.
 */

#pragma once

#include "plssvm/backends/OpenMP/csvm.hpp"  // plssvm::openmp::csvm
#include "plssvm/kernel_types.hpp"          // plssvm::kernel_type
#include "plssvm/parameter.hpp"             // plssvm::parameter

#include <vector>  // std::vector

/**
 * @brief GTest mock class for the OpenMP CSVM.
 * @tparam T the type of the data
 */
template <typename T>
class mock_openmp_csvm : public plssvm::openmp::csvm<T> {
    using base_type = plssvm::openmp::csvm<T>;

  public:
    using real_type = typename base_type::real_type;
    using size_type = typename base_type::size_type;

    explicit mock_openmp_csvm(const plssvm::parameter<T> &params) :
        base_type{ params } {}
    explicit mock_openmp_csvm(const plssvm::kernel_type kernel, const real_type degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
        base_type{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {}

    // make non-virtual functions publicly visible
    using base_type::generate_q;
    using base_type::learn;
    using base_type::run_device_kernel;
    using base_type::setup_data_on_device;

    // parameter setter
    void set_cost(const real_type cost) { cost_ = cost; }
    void set_QA_cost(const real_type QA_cost) { QA_cost_ = QA_cost; }

    // getter for internal variables
    const std::vector<std::vector<real_type>> &get_device_data() const { return data_; }

  private:
    using base_type::cost_;
    using base_type::QA_cost_;

    using base_type::data_;
};
