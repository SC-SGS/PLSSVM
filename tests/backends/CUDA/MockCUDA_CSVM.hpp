#ifndef TESTS_BACKENDS_CUDA_TEST
#define TESTS_BACKENDS_CUDA_TEST
#include "plssvm/CSVM.hpp"

#include "plssvm/backends/CUDA/CUDA_CSVM.hpp"

#include "plssvm/typedef.hpp"
#include "gmock/gmock.h"
#include <plssvm/kernel_types.hpp>

class MockCUDA_CSVM : public plssvm::CUDA_CSVM {
  public:
    MockCUDA_CSVM(real_t cost_ = 1.5, real_t epsilon_ = 0.00001, plssvm::kernel_type kernel_ = plssvm::kernel_type::linear, real_t degree_ = 2.0, real_t gamma_ = 1.5, real_t coef0_ = 0.0, bool info_ = false) :
        plssvm::CUDA_CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}
    // MOCK_METHOD(void, load_w, (), (override));
    MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, (), (override));
    using plssvm::CUDA_CSVM::alpha;
    using plssvm::CUDA_CSVM::bias;
    using plssvm::CUDA_CSVM::CG;
    using plssvm::CUDA_CSVM::generate_q;
    using plssvm::CUDA_CSVM::kernel_function;
    using plssvm::CUDA_CSVM::learn;
    using plssvm::CUDA_CSVM::loadDataDevice;
    using plssvm::CUDA_CSVM::QA_cost;

    const real_t get_num_data_points() const {
        return num_data_points;
    }
    const real_t get_num_features() const {
        return num_features;
    }
    std::vector<std::vector<real_t>> get_data() const {
        return data;
    }
    const real_t get_gamma() const {
        return gamma;
    }
};

#endif /* TESTS_BACKENDS_CUDA_TEST */
