/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief MOCK class for the C-SVM base class.
 */

#pragma once

#include "plssvm/csvm.hpp"          // plssvm::csvm
#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type
#include "plssvm/parameter.hpp"     // plssvm::parameter

#include "gmock/gmock.h"  // MOCK_METHOD

#include <vector>  // std::vector

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

    explicit mock_csvm(const plssvm::parameter<T> &params) :
        base_type{ params } {}
    explicit mock_csvm(const plssvm::kernel_type kernel, const int degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
        base_type{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {}

    // mock pure virtual functions
    MOCK_METHOD(void, setup_data_on_device, (), (override));
    MOCK_METHOD(std::vector<real_type>, generate_q, (), (override));
    MOCK_METHOD(std::vector<real_type>, solver_CG, (const std::vector<real_type> &b, const size_type, const real_type, const std::vector<real_type> &), (override));

    // make non-virtual functions publicly visible
    using base_type::kernel_function;
    using base_type::transform_data;

    // parameter getter
    int get_degree() const { return degree_; }
    real_type get_gamma() const { return gamma_; }
    real_type get_coef0() const { return coef0_; }
    real_type get_cost() const { return cost_; }
    using base_type::epsilon_;
    using base_type::kernel_;
    using base_type::print_info_;

    // getter for internal variables
    size_type get_num_data_points() const { return num_data_points_; }
    size_type get_num_features() const { return num_features_; }
    const std::vector<std::vector<real_type>> &get_data() const { return *data_ptr_; }

  private:
    // make template ase variables visible
    using base_type::coef0_;
    using base_type::cost_;
    using base_type::data_ptr_;
    using base_type::degree_;
    using base_type::gamma_;
    using base_type::num_data_points_;
    using base_type::num_features_;
};
