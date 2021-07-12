#ifndef TESTS_BACKENDS_CUDA_TEST
#define TESTS_BACKENDS_CUDA_TEST

#include "plssvm/CSVM.hpp"

#include "plssvm/backends/CUDA/CUDA_CSVM.hpp"

#include "plssvm/typedef.hpp"
#include "gmock/gmock.h"
#include <plssvm/kernel_types.hpp>

class MockCUDA_CSVM : public plssvm::CUDA_CSVM<double> {
    using base_type = plssvm::CUDA_CSVM<double>;

  public:
    using base_type::real_type;
    using base_type::size_type;

    explicit MockCUDA_CSVM(plssvm::kernel_type kernel_ = plssvm::kernel_type::linear, real_type degree_ = 2.0, real_type gamma_ = 1.5, real_type coef0_ = 0.0, real_type cost_ = 1.5, real_type epsilon_ = 0.00001, bool info_ = false) :
        base_type{ kernel_, degree_, gamma_, coef0_, cost_, epsilon_, info_ } {}
    MOCK_METHOD(void, load_w, (), (override));
    //    MOCK_METHOD(std::vector<real_type>, predict, (real_type *, size_type, size_type));
    MOCK_METHOD(void, learn, ());
    using base_type::alpha_;
    using base_type::bias_;
    using base_type::data_;
    using base_type::gamma_;
    using base_type::generate_q;
    using base_type::kernel_function;
    using base_type::learn;
    using base_type::num_data_points_;
    using base_type::num_features_;
    using base_type::QA_cost_;
    using base_type::setup_data_on_device;
    using base_type::solver_CG;

    size_type get_num_data_points() const {
        return num_data_points_;
    }
    size_type get_num_features() const {
        return num_features_;
    }
    std::vector<std::vector<real_type>> get_data() const {
        return data_;
    }
    real_type get_gamma() const {
        return gamma_;
    }
};

#endif /* TESTS_BACKENDS_CUDA_TEST */
