#include "../MockCSVM.hpp"
#include "../compare.hpp"
#include "MockOpenMP_CSVM.hpp"
#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#include "plssvm/backends/OpenMP/svm-kernel.hpp"
#include "plssvm/kernel_types.hpp"

#include <gtest/gtest.h>

#include <fstream>
#include <random>

TEST(OpenMP, writeModel) {
    std::string model = std::tmpnam(nullptr);
    MockOpenMP_CSVM csvm2(plssvm::kernel_type::linear, 3.0, 0.0, 0.0, 1., 0.001, false);
    std::string testfile = TESTPATH "/data/5x4.libsvm";
    csvm2.learn(testfile, model);

    std::ifstream model_ifs(model);
    std::string genfile2((std::istreambuf_iterator<char>(model_ifs)),
                         std::istreambuf_iterator<char>());
    remove(model.c_str());

    EXPECT_THAT(genfile2, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv [1-9][0-9]*\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
}

TEST(OpenMP, q) {
    MockOpenMP_CSVM csvm_OpenMP;
    using real_type = typename MockOpenMP_CSVM::real_type;

    std::vector correct = generate_q<real_type>(TESTFILE);

    csvm_OpenMP.parse_libsvm(TESTFILE);
    csvm_OpenMP.setup_data_on_device();
    std::vector test = csvm_OpenMP.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], test[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TEST(OpenMP, linear) {
    MockOpenMP_CSVM csvm_OpenMP(plssvm::kernel_type::linear);
    using real_type = typename MockOpenMP_CSVM::real_type;

    const size_t size = 512;
    std::vector<real_type> x1(size);
    std::vector<real_type> x2(size);
    std::generate(x1.begin(), x1.end(), std::rand);
    std::generate(x2.begin(), x2.end(), std::rand);
    real_type correct = linear_kernel(x1, x2);

    real_type result_OpenMP = csvm_OpenMP.kernel_function(x1, x2);
    real_type result2_OpenMP = csvm_OpenMP.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_OpenMP);
    EXPECT_DOUBLE_EQ(correct, result2_OpenMP);
}

TEST(OpenMP, q_linear) {
    MockCSVM csvm;
    using real_type = typename MockCSVM::real_type;

    csvm.parse_libsvm(TESTFILE);
    std::vector<real_type> correct = q<plssvm::kernel_type::linear>(csvm.get_data());

    MockOpenMP_CSVM csvm_OpenMP(plssvm::kernel_type::linear);
    csvm_OpenMP.parse_libsvm(TESTFILE);
    csvm_OpenMP.setup_data_on_device();
    std::vector<real_type> test = csvm_OpenMP.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], test[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TEST(OpenMP, kernel_linear) {
    MockCSVM csvm;
    using real_type = typename MockCSVM::real_type;

    csvm.parse_libsvm(TESTFILE);

    const size_t dept = csvm.get_num_data_points() - 1;

    std::vector<real_type> x(dept);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(-1, 2.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

    const std::vector<real_type> q_ = q<plssvm::kernel_type::linear>(csvm.get_data());

    const real_type cost = csvm.cost_;

    const real_type QA_cost = linear_kernel(csvm.data_.back(), csvm.data_.back()) + 1 / cost;

    for (const real_type sgn : { -1.0, 1.0 }) {
        std::vector<real_type> correct = kernel_linear_function(csvm.get_data(), x, q_, sgn, QA_cost, cost);

        std::vector<real_type> result(dept, 0.0);
        plssvm::device_kernel_linear(csvm.data_, result, x, QA_cost, 1 / cost, sgn);

        ASSERT_EQ(correct.size(), result.size()) << "sgn: " << sgn;
        for (size_t index = 0; index < correct.size(); ++index) {
            EXPECT_NEAR(correct[index], result[index], std::abs(correct[index] * 1e-10)) << " index: " << index << " sgn: " << sgn;
        }
    }
}