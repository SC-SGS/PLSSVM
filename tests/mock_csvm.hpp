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

#include "plssvm/csvm.hpp"          // plssvm::csvm
#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type
#include "plssvm/parameter.hpp"     // plssvm::parameter

#include "gmock/gmock.h"  // MOCK_METHOD

#include <utility>  // std::pair
#include <vector>   // std::vector

/**
 * @brief GTest mock class for the base CSVM class.
 * @tparam T the type of the data points
 */
template <typename T>
class mock_csvm final : public plssvm::csvm<T> {
    using base_type = plssvm::csvm<T>;

  public:
    using real_type = typename base_type::real_type;
    using size_type = typename base_type::size_type;

    explicit mock_csvm(plssvm::parameter<T> params = {}) :
        base_type{ std::move(params) } {}
    template <typename... Args>
    explicit mock_csvm(const plssvm::kernel_type kernel, Args &&...args) :
        base_type{ kernel, std::forward<Args>(args)... } {}

    // mock pure virtual functions
    MOCK_METHOD((std::pair<std::vector<real_type>, real_type>), solve_system_of_linear_equations, (const plssvm::parameter<real_type> &, const std::vector<std::vector<real_type>> &, std::vector<real_type>, real_type, size_type), (const, override));
    MOCK_METHOD(std::vector<real_type>, predict_values, (const plssvm::parameter<real_type> &, const std::vector<std::vector<real_type>> &, const std::vector<real_type> &, real_type, std::vector<real_type> &, const std::vector<std::vector<real_type>> &), (const, override));
};

#endif  // PLSSVM_TESTS_MOCK_CSVM_HPP_