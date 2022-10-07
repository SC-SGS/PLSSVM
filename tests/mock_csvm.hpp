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

#pragma once

#include "plssvm/csvm.hpp"              // plssvm::csvm
#include "plssvm/kernel_types.hpp"      // plssvm::kernel_type
#include "plssvm/parameter.hpp"         // plssvm::parameter
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "gmock/gmock.h"  // MOCK_METHOD

#include <memory>   // std::shared_ptr
#include <utility>  // std::pair
#include <vector>   // std::vector

/**
 * @brief GTest mock class for the base CSVM class.
 * @tparam T the type of the data
 */
template <typename T>
class mock_csvm : public plssvm::csvm<T> {
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

//    // make non-virtual functions publicly visible
//    using base_type::kernel_function;
//    using base_type::predict;  // no idea way necessary (since the used 'real_type predict(const std::vector<real_type>&)' is a public member function) but it works
//    using base_type::transform_data;

    // getter for all parameter
//    plssvm::target_platform get_target() const { return base_type::target_; }
//    plssvm::kernel_type get_kernel() const { return base_type::kernel_; }
//    int get_degree() const { return base_type::degree_; }
//    real_type get_gamma() const { return base_type::gamma_; }
//    real_type get_coef0() const { return base_type::coef0_; }
//    real_type get_cost() const { return base_type::cost_; }
//    real_type get_epsilon() const { return base_type::epsilon_; }
//    bool get_print_info() const { return base_type::print_info_; }
//
//    const std::shared_ptr<const std::vector<std::vector<real_type>>> &get_data_ptr() const { return base_type::data_ptr_; }
//    const std::vector<std::vector<real_type>> &get_data() const { return *base_type::data_ptr_; }
//    std::shared_ptr<const std::vector<real_type>> &get_value_ptr() { return base_type::value_ptr_; }
//    std::shared_ptr<const std::vector<real_type>> &get_alpha_ptr() { return base_type::alpha_ptr_; }
//
//    std::size_t get_num_data_points() const { return base_type::num_data_points_; }
//    std::size_t get_num_features() const { return base_type::num_features_; }
//    real_type get_bias() const { return base_type::bias_; }
//    real_type get_QA_cost() const { return base_type::QA_cost_; }
};
