#ifndef TESTS_BACKENDS_OPENCL_MOCKOPENCL_CSVM
#define TESTS_BACKENDS_OPENCL_MOCKOPENCL_CSVM
#include "plssvm/CSVM.hpp"

#include "plssvm/backends/OpenCL/OpenCL_CSVM.hpp"

#include "plssvm/typedef.hpp"
#include "gmock/gmock.h"
#include <plssvm/kernel_types.hpp>

class MockOpenCL_CSVM : public plssvm::OpenCL_CSVM {
  public:
    MockOpenCL_CSVM(real_t cost_ = 1.5, real_t epsilon_ = 0.00001, plssvm::kernel_type kernel_ = plssvm::kernel_type::linear, real_t degree_ = 2.0, real_t gamma_ = 1.5, real_t coef0_ = 0.0, bool info_ = false) :
        plssvm::OpenCL_CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}
    // MOCK_METHOD(void, load_w, (), (override));
    MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, (), (override));
    using plssvm::OpenCL_CSVM::alpha;
    using plssvm::OpenCL_CSVM::bias;
    using plssvm::OpenCL_CSVM::QA_cost;

    using plssvm::OpenCL_CSVM::CG;
    using plssvm::OpenCL_CSVM::learn;
    using plssvm::OpenCL_CSVM::loadDataDevice;

    using plssvm::OpenCL_CSVM::cost;
    using plssvm::OpenCL_CSVM::data;
    using plssvm::OpenCL_CSVM::data_cl;
    using plssvm::OpenCL_CSVM::generate_q;
    using plssvm::OpenCL_CSVM::kernel_function;
    using plssvm::OpenCL_CSVM::manager;

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

#endif /* TESTS_BACKENDS_OPENCL_MOCKOPENCL_CSVM */
