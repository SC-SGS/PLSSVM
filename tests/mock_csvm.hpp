/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM base class.
 */

#ifndef PLSSVM_TESTS_MOCK_CSVM_HPP_
#define PLSSVM_TESTS_MOCK_CSVM_HPP_
#pragma once

#include "plssvm/csvm.hpp"                   // plssvm::csvm
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "gmock/gmock.h"  // MOCK_METHOD

#include <utility>  // std::pair
#include <vector>   // std::vector

/**
 * @brief GTest mock class for the base CSVM class.
 * @tparam T the type of the data points
 */
class mock_csvm final : public plssvm::csvm {
  public:
    explicit mock_csvm(plssvm::parameter params = {}) :
        plssvm::csvm{ params } {}
    template <typename... Args>
    explicit mock_csvm(const plssvm::kernel_function_type kernel, Args &&...args) :
        plssvm::csvm{ kernel, std::forward<Args>(args)... } {}

    // mock pure virtual functions
    MOCK_METHOD((std::pair<std::vector<float>, float>), solve_system_of_linear_equations, (const plssvm::detail::parameter<float> &, const std::vector<std::vector<float>> &, std::vector<float>, float, unsigned long long), (const, override));
    MOCK_METHOD((std::pair<std::vector<double>, double>), solve_system_of_linear_equations, (const plssvm::detail::parameter<double> &, const std::vector<std::vector<double>> &, std::vector<double>, double, unsigned long long), (const, override));
    MOCK_METHOD(std::vector<float>, predict_values, (const plssvm::detail::parameter<float> &, const std::vector<std::vector<float>> &, const std::vector<float> &, float, std::vector<float> &, const std::vector<std::vector<float>> &), (const, override));
    MOCK_METHOD(std::vector<double>, predict_values, (const plssvm::detail::parameter<double> &, const std::vector<std::vector<double>> &, const std::vector<double> &, double, std::vector<double> &, const std::vector<std::vector<double>> &), (const, override));
};

#endif  // PLSSVM_TESTS_MOCK_CSVM_HPP_