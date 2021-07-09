#pragma once
#include "plssvm/CSVM.hpp"
#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    #include "plssvm/backends/OpenCL/OpenCL_CSVM.hpp"
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    #include "plssvm/backends/CUDA/CUDA_CSVM.hpp"
#endif
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    #include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#endif
#include "plssvm/parameter.hpp"
#include "plssvm/typedef.hpp"
#include "gmock/gmock.h"
#include <plssvm/kernel_types.hpp>

using real_t = plssvm::real_t;  // TODO:

class MockCSVM : public plssvm::CSVM<real_t> {
  public:
    explicit MockCSVM(plssvm::parameter<real_t> &params) :
        plssvm::CSVM<real_t>{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}
    MockCSVM(plssvm::kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info) :
        plssvm::CSVM<real_t>{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {}

    // MOCK_METHOD(void, load_w, (), (override));
    // MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, ());
    MOCK_METHOD(void, setup_data_on_device, (), (override));
    MOCK_METHOD(std::vector<real_t>, generate_q, (), (override));
    MOCK_METHOD(std::vector<real_t>, solver_CG, (const std::vector<real_t> &b, std::size_t, real_t, const std::vector<real_t> &), (override));
    using plssvm::CSVM<real_t>::cost_;
    using plssvm::CSVM<real_t>::data_;
    using plssvm::CSVM<real_t>::kernel_function;
    using plssvm::CSVM<real_t>::transform_data;
    using plssvm::CSVM<real_t>::value_;

    const real_t get_num_data_points() const { return num_data_points_; }
    const real_t get_num_features() const { return num_features_; }
    std::vector<std::vector<real_t>> get_data() const { return data_; }
    const real_t get_gamma() const { return gamma_; }
};
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
class MockOpenMP_CSVM : public plssvm::OpenMP_CSVM<real_t> {
  public:
    explicit MockOpenMP_CSVM(plssvm::parameter<real_t> &params) :
        plssvm::OpenMP_CSVM<real_t>{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}
    MockOpenMP_CSVM(plssvm::kernel_type kernel, real_t degree, real_t gamma, real_t coef0, real_t cost, real_t epsilon, bool print_info) :
        plssvm::OpenMP_CSVM<real_t>{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {}

    // MOCK_METHOD(void, load_w, (), (override));
    // MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, ());
    using plssvm::OpenMP_CSVM<real_t>::alpha_;
    using plssvm::OpenMP_CSVM<real_t>::bias_;
    using plssvm::OpenMP_CSVM<real_t>::QA_cost_;

    using plssvm::OpenMP_CSVM<real_t>::learn;
    using plssvm::OpenMP_CSVM<real_t>::setup_data_on_device;
    using plssvm::OpenMP_CSVM<real_t>::solver_CG;

    using plssvm::OpenMP_CSVM<real_t>::generate_q;
    using plssvm::OpenMP_CSVM<real_t>::kernel_function;

    const real_t get_num_data_points() const {
        return num_data_points_;
    }
    const real_t get_num_features() const {
        return num_features_;
    }
    std::vector<std::vector<real_t>> get_data() const {
        return data_;
    }
    const real_t get_gamma() const {
        return gamma_;
    }
};
#else
    #pragma message("Ignore OpenMP backend test")
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
class MockOpenCL_CSVM : public plssvm::OpenCL_CSVM<real_t> {
  public:
    explicit MockOpenCL_CSVM(plssvm::parameter<real_t> &params) :
        plssvm::OpenCL_CSVM<real_t>{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}
    MockOpenCL_CSVM(plssvm::kernel_type kernel, real_t degree, real_t gamma, real_t coef0, real_t cost, real_t epsilon, bool print_info) :
        plssvm::OpenCL_CSVM<real_t>{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {}

    // MOCK_METHOD(void, load_w, (), (override));
    // MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, ());
    using plssvm::OpenCL_CSVM<real_t>::alpha_;
    using plssvm::OpenCL_CSVM<real_t>::bias_;
    using plssvm::OpenCL_CSVM<real_t>::QA_cost_;

    using plssvm::OpenCL_CSVM<real_t>::learn;
    using plssvm::OpenCL_CSVM<real_t>::setup_data_on_device;
    using plssvm::OpenCL_CSVM<real_t>::solver_CG;

    using plssvm::OpenCL_CSVM<real_t>::cost_;
    using plssvm::OpenCL_CSVM<real_t>::data_;
    using plssvm::OpenCL_CSVM<real_t>::data_cl;
    using plssvm::OpenCL_CSVM<real_t>::generate_q;
    using plssvm::OpenCL_CSVM<real_t>::kernel_function;
    using plssvm::OpenCL_CSVM<real_t>::manager;

    const real_t get_num_data_points() const {
        return num_data_points_;
    }
    const real_t get_num_features() const {
        return num_features_;
    }
    std::vector<std::vector<real_t>> get_data() const {
        return data_;
    }
    const real_t get_gamma() const {
        return gamma_;
    }
};
#else
    #pragma message("Ignore OpenCL backend Test")
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
class MockCUDA_CSVM : public plssvm::CUDA_CSVM<real_t> {
  public:
    explicit MockCUDA_CSVM(plssvm::parameter<real_t> &params) :
        plssvm::CUDA_CSVM<real_t>{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}
    MockCUDA_CSVM(plssvm::kernel_type kernel, real_t degree, real_t gamma, real_t coef0, real_t cost, real_t epsilon, bool print_info) :
        plssvm::CUDA_CSVM<real_t>{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {}

    // MOCK_METHOD(void, load_w, (), (override));
    // MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, ());
    using plssvm::CUDA_CSVM<real_t>::alpha_;
    using plssvm::CUDA_CSVM<real_t>::bias_;
    using plssvm::CUDA_CSVM<real_t>::generate_q;
    using plssvm::CUDA_CSVM<real_t>::kernel_function;
    using plssvm::CUDA_CSVM<real_t>::learn;
    using plssvm::CUDA_CSVM<real_t>::QA_cost_;
    using plssvm::CUDA_CSVM<real_t>::setup_data_on_device;
    using plssvm::CUDA_CSVM<real_t>::solver_CG;

    const real_t get_num_data_points() const {
        return num_data_points_;
    }
    const real_t get_num_features() const {
        return num_features_;
    }
    std::vector<std::vector<real_t>> get_data() const {
        return data_;
    }
    const real_t get_gamma() const {
        return gamma_;
    }
};
#else
    #pragma message("Ignore CUDA backend Test")
#endif
