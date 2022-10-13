/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM class using the OpenMP backend.
 */

#ifndef PLSSVM_TESTS_BACKENDS_OPENMP_MOCK_OPENMP_CSVM_HPP_
#define PLSSVM_TESTS_BACKENDS_OPENMP_MOCK_OPENMP_CSVM_HPP_
#pragma once

#include "plssvm/backends/OpenMP/csvm.hpp"   // plssvm::openmp::csvm
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"              // plssvm::parameter
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

#include <vector>  // std::vector

/**
 * @brief GTest mock class for the OpenMP CSVM.
 * @tparam T the type of the data
 */
class mock_openmp_csvm final : public plssvm::openmp::csvm {
    using base_type = plssvm::openmp::csvm;

  public:
    using size_type = typename base_type::size_type;

    explicit mock_openmp_csvm(plssvm::target_platform target, plssvm::parameter params = {}) :
        base_type{ target, params } {}
    template <typename... Args>
    mock_openmp_csvm(plssvm::target_platform target, plssvm::kernel_function_type kernel, Args &&...named_args) :
        base_type{ target, kernel, std::forward<Args>(named_args)... } {}

    //    // make non-virtual functions publicly visible
    //    using base_type::generate_q;
    //    using base_type::run_device_kernel;
    //    using base_type::setup_data_on_device;
    //
    //    // parameter setter
    //    void set_cost(const real_type cost) { base_type::cost_ = cost; }
    //    void set_QA_cost(const real_type QA_cost) { base_type::QA_cost_ = QA_cost; }
    //
    //    // getter for internal variable
    //    std::shared_ptr<const std::vector<real_type>> &get_alpha_ptr() { return base_type::alpha_ptr_; }
    //    const std::vector<std::vector<real_type>> &get_device_data() const { return *base_type::data_ptr_; }
    //    std::size_t get_num_devices() const { return 1; }
};

#endif  // PLSSVM_TESTS_BACKENDS_OPENMP_MOCK_OPENMP_CSVM_HPP_