/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief MOCK class for the C-SVM class using the OpenCL backend.
 */
#pragma once

#include "plssvm/backends/OpenCL/OpenCL_CSVM.hpp"  // plssvm::OpenCL_CSVM
#include "plssvm/kernel_types.hpp"                 // plssvm::kernel_type
#include "plssvm/parameter.hpp"                    // plssvm::parameter

#include "gmock/gmock.h"  // MOCK_METHOD

template <typename T>
class MockOpenCL_CSVM : public plssvm::OpenCL_CSVM<T> {
    using base_type = plssvm::OpenCL_CSVM<T>;

  public:
    using real_type = typename base_type::real_type;
    using size_type = typename base_type::size_type;

    explicit MockOpenCL_CSVM(const plssvm::parameter<T> &params) :
        base_type{ params } {}
    explicit MockOpenCL_CSVM(plssvm::kernel_type kernel_ = plssvm::kernel_type::linear, real_type degree_ = 2.0, real_type gamma_ = 1.5, real_type coef0_ = 0.0, real_type cost_ = 1.5, real_type epsilon_ = 0.00001, bool info_ = false) :
        base_type{ kernel_, degree_, gamma_, coef0_, cost_, epsilon_, info_ } {}

    MOCK_METHOD(void, load_w, (), (override));
    // MOCK_METHOD(std::vector<real_type>, predict, (real_type *, size_type, size_type), (override));
    MOCK_METHOD(void, learn, ());

    using base_type::QA_cost_;

    using base_type::generate_q;
    using base_type::kernel_function;
    using base_type::learn;
    using base_type::manager;
    using base_type::setup_data_on_device;
    using base_type::solver_CG;

    using base_type::cost_;
    using base_type::data_cl;

    size_type get_num_data_points() const { return num_data_points_; }
    size_type get_num_features() const { return num_features_; }
    const std::vector<std::vector<real_type>> &get_data() const { return data_; }
    real_type get_gamma() const { return gamma_; }

  private:
    using base_type::data_;
    using base_type::gamma_;
    using base_type::num_data_points_;
    using base_type::num_features_;
};
