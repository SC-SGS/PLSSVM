#ifndef TESTS_MOCKCSVM
#define TESTS_MOCKCSVM
#include "plssvm/CSVM.hpp"
#include "plssvm/typedef.hpp"
#include "gmock/gmock.h"
#include <plssvm/kernel_types.hpp>

class MockCSVM : public plssvm::CSVM {
  public:
    MockCSVM(real_t cost_ = 1.5, real_t epsilon_ = 0.00001, plssvm::kernel_type kernel_ = plssvm::kernel_type::linear, real_t degree_ = 2.0, real_t gamma_ = 1.5, real_t coef0_ = 0.0, bool info_ = false) :
        plssvm::CSVM(cost_, epsilon_, kernel_, degree_, gamma_, coef0_, info_) {}
    MOCK_METHOD(void, load_w, (), (override));
    MOCK_METHOD(std::vector<real_t>, predict, (real_t *, int, int), (override));
    MOCK_METHOD(void, learn, (), (override));
    MOCK_METHOD(void, loadDataDevice, (), (override));
    MOCK_METHOD(std::vector<real_t>, generate_q, (), (override));
    MOCK_METHOD(std::vector<real_t>, CG, (const std::vector<real_t> &b, const int, const real_t, const std::vector<real_t> &), (override));
    using plssvm::CSVM::cost;
    using plssvm::CSVM::data;
    using plssvm::CSVM::kernel_function;
    using plssvm::CSVM::transform_data;
    using plssvm::CSVM::value;

    const real_t get_num_data_points() const { return num_data_points; }
    const real_t get_num_features() const { return num_features; }
    std::vector<std::vector<real_t>> get_data() const { return data; }
    const real_t get_gamma() const { return gamma; }
};

#endif /* TESTS_MOCKCSVM */
