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

#include "plssvm/backends/OpenMP/csvm.hpp"  // plssvm::openmp::csvm
#include "plssvm/parameter.hpp"             // plssvm::parameter

/**
 * @brief GTest mock class for the OpenMP CSVM.
 */
class mock_openmp_csvm final : public plssvm::openmp::csvm {
    using base_type = plssvm::openmp::csvm;

  public:
    template <typename... Args>
    explicit mock_openmp_csvm(Args&&... args) :
        base_type{ std::forward<Args>(args)... } {}

    // make protected member functions public
    using base_type::calculate_w;
    using base_type::generate_q;
    using base_type::predict_values;
    using base_type::run_device_kernel;
    using base_type::solve_system_of_linear_equations;
};

#endif  // PLSSVM_TESTS_BACKENDS_OPENMP_MOCK_OPENMP_CSVM_HPP_