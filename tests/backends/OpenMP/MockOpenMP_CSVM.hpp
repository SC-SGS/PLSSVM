#ifndef TESTS_BACKENDS_OPENMP_MOCKOPENMP_CSVM
#define TESTS_BACKENDS_OPENMP_MOCKOPENMP_CSVM

#include "plssvm/CSVM.hpp"

#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"

#include "plssvm/typedef.hpp"
#include "gmock/gmock.h"
#include <plssvm/kernel_types.hpp>

class MockOpenMP_CSVM : public plssvm::OpenMP_CSVM {
  public:
    MockOpenMP_CSVM(real_t cost_ = 1.5, real_t epsilon_ = 0.00001, plssvm::kernel_type kernel_ = plssvm::kernel_type::linear, real_t degree_ = 2.0, real_t gamma_ = 1.5, real_t coef0_ = 0.0, bool info_ = false) :
        plssvm::OpenMP_CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}
    // MOCK_METHOD(void, load_w, (), (override));
    MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    // MOCK_METHOD(void, learn, (), (override));
    using plssvm::OpenMP_CSVM::alpha;
    using plssvm::OpenMP_CSVM::bias;
    using plssvm::OpenMP_CSVM::QA_cost;

    using plssvm::OpenMP_CSVM::CG;
    using plssvm::OpenMP_CSVM::learn;
    using plssvm::OpenMP_CSVM::loadDataDevice;

    using plssvm::OpenMP_CSVM::generate_q;
    using plssvm::OpenMP_CSVM::kernel_function;

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

#endif /* TESTS_BACKENDS_OPENMP_MOCKOPENMP_CSVM */
