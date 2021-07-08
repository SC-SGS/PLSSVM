#include "../MockCSVM.hpp"
#include "../compare.hpp"
#include "MockOpenMP_CSVM.hpp"
#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#include "plssvm/backends/OpenMP/svm-kernel.hpp"
#include "plssvm/kernel_types.hpp"

#include <gtest/gtest.h>

#include <random>

TEST(IO, writeModel) {
    std::string model = std::tmpnam(nullptr);
    MockOpenMP_CSVM csvm2(1., 0.001, plssvm::kernel_type::linear, 3.0, 0.0, 0.0, false);
    std::string testfile = TESTPATH "/data/5x4.libsvm";
    csvm2.learn(testfile, model);

    std::ifstream model_ifs(model);
    std::string genfile2((std::istreambuf_iterator<char>(model_ifs)),
                         std::istreambuf_iterator<char>());
    remove(model.c_str());

    EXPECT_THAT(genfile2, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv [1-9][0-9]*\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
}

TEST(learn, q) {
    std::vector correct = generate_q(TESTFILE);

    MockOpenMP_CSVM csvm_OpenMP;
    csvm_OpenMP.libsvmParser(TESTFILE);
    csvm_OpenMP.loadDataDevice();
    std::vector test = csvm_OpenMP.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_DOUBLE_EQ(correct[index], test[index]) << " index: " << index;
    }
}

TEST(kernel, linear) {
    const size_t size = 512;
    std::vector<real_t> x1(size);
    std::vector<real_t> x2(size);
    std::generate(x1.begin(), x1.end(), std::rand);
    std::generate(x2.begin(), x2.end(), std::rand);
    real_t correct = linear_kernel(x1, x2);

    MockOpenMP_CSVM csvm_OpenMP(1., 0.001, plssvm::kernel_type::linear);
    real_t result_OpenMP = csvm_OpenMP.kernel_function(x1, x2);
    real_t result2_OpenMP = csvm_OpenMP.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_OpenMP);
    EXPECT_DOUBLE_EQ(correct, result2_OpenMP);
}

TEST(learn, q_linear) {
    MockCSVM csvm;
    csvm.libsvmParser(TESTFILE);
    std::vector<real_t> correct = q<plssvm::kernel_type::linear>(csvm.get_data());

    MockOpenMP_CSVM csvm_OpenMP(1., 0.001, plssvm::kernel_type::linear);
    csvm_OpenMP.libsvmParser(TESTFILE);
    csvm_OpenMP.loadDataDevice();
    std::vector<real_t> test = csvm_OpenMP.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_DOUBLE_EQ(correct[index], test[index]) << " index: " << index;
    }
}

TEST(learn, kernel_linear) {
    MockCSVM csvm;
    csvm.libsvmParser(TESTFILE);

    const size_t dept = csvm.get_num_data_points() - 1;

    std::vector<real_t> x(dept);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_t> dist(-1, 2.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

    const std::vector<real_t> q_ = q<plssvm::kernel_type::linear>(csvm.get_data());

    const real_t cost = csvm.cost;

    const real_t QA_cost = linear_kernel(csvm.data.back(), csvm.data.back()) + 1 / cost;

    for (const real_t sgn : { -1.0, 1.0 }) {
        std::vector<real_t> correct = kernel_linear_function(csvm.get_data(), x, q_, sgn, QA_cost, cost);

        std::vector<real_t> result(dept, 0.0);
        plssvm::kernel_linear(result, csvm.data, &csvm.data.back()[0], q_.data(), result, x.data(), csvm.get_num_features(), QA_cost, 1 / cost, sgn);

        ASSERT_EQ(correct.size(), result.size()) << "sgn: " << sgn;
        for (size_t index = 0; index < correct.size(); ++index) {
            EXPECT_NEAR(correct[index], result[index], 1e-8) << " index: " << index << " sgn: " << sgn;
        }
    }
}