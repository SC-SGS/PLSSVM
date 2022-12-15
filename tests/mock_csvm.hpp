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
#include "plssvm/parameter.hpp"              // plssvm::parameter, plssvm::detail::parameter

#include "gmock/gmock.h"  // MOCK_METHOD, ON_CALL, ::testing::{An, Return}

#include <utility>  // std::pair, std::forward
#include <vector>   // std::vector

template <typename real_type>
const std::pair<std::vector<real_type>, real_type> solve_system_of_linear_equations_fake_return{ { real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 }, real_type{ 4.0 }, real_type{ 5.0 } }, real_type{ 3.1415 } };

template <typename real_type>
const std::vector<real_type> predict_values_fake_return{ real_type{ -1.0 }, real_type{ -1.2 }, real_type{ -0.5 }, real_type{ 1.0 }, real_type{ 2.4 } };

/**
 * @brief GTest mock class for the base CSVM class.
 */
class mock_csvm final : public plssvm::csvm {
  public:
    explicit mock_csvm(plssvm::parameter params = {}) :
        plssvm::csvm{ params } {
        this->fake_functions();
    }
    template <typename... Args>
    explicit mock_csvm(Args &&...args) :
        plssvm::csvm{ std::forward<Args>(args)... } {
        this->fake_functions();
    }

    // mock pure virtual functions
    MOCK_METHOD((std::pair<std::vector<float>, float>), solve_system_of_linear_equations, (const plssvm::detail::parameter<float> &, const std::vector<std::vector<float>> &, std::vector<float>, float, unsigned long long), (const, override));
    MOCK_METHOD((std::pair<std::vector<double>, double>), solve_system_of_linear_equations, (const plssvm::detail::parameter<double> &, const std::vector<std::vector<double>> &, std::vector<double>, double, unsigned long long), (const, override));
    MOCK_METHOD(std::vector<float>, predict_values, (const plssvm::detail::parameter<float> &, const std::vector<std::vector<float>> &, const std::vector<float> &, float, std::vector<float> &, const std::vector<std::vector<float>> &), (const, override));
    MOCK_METHOD(std::vector<double>, predict_values, (const plssvm::detail::parameter<double> &, const std::vector<std::vector<double>> &, const std::vector<double> &, double, std::vector<double> &, const std::vector<std::vector<double>> &), (const, override));

  private:
    void fake_functions() const {
        // clang-format off
        ON_CALL(*this, solve_system_of_linear_equations(
                           ::testing::An<const plssvm::detail::parameter<float> &>(),
                           ::testing::An<const std::vector<std::vector<float>> &>(),
                           ::testing::An<std::vector<float>>(),
                           ::testing::An<float>(),
                           ::testing::An<unsigned long long>())).WillByDefault(::testing::Return(solve_system_of_linear_equations_fake_return<float>));

        ON_CALL(*this, solve_system_of_linear_equations(
                           ::testing::An<const plssvm::detail::parameter<double> &>(),
                           ::testing::An<const std::vector<std::vector<double>> &>(),
                           ::testing::An<std::vector<double>>(),
                           ::testing::An<double>(),
                           ::testing::An<unsigned long long>())).WillByDefault(::testing::Return(solve_system_of_linear_equations_fake_return<double>));

        ON_CALL(*this, predict_values(
                           ::testing::An<const plssvm::detail::parameter<float> &>(),
                           ::testing::An<const std::vector<std::vector<float>> &>(),
                           ::testing::An<const std::vector<float> &>(),
                           ::testing::An<float>(),
                           ::testing::An<std::vector<float> &>(),
                           ::testing::An<const std::vector<std::vector<float>> &>())).WillByDefault(::testing::Return(predict_values_fake_return<float>));

        ON_CALL(*this, predict_values(
                           ::testing::An<const plssvm::detail::parameter<double> &>(),
                           ::testing::An<const std::vector<std::vector<double>> &>(),
                           ::testing::An<const std::vector<double> &>(),
                           ::testing::An<double>(),
                           ::testing::An<std::vector<double> &>(),
                           ::testing::An<const std::vector<std::vector<double>> &>())).WillByDefault(::testing::Return(predict_values_fake_return<double>));
        // clang-format on
    }
};

#endif  // PLSSVM_TESTS_MOCK_CSVM_HPP_