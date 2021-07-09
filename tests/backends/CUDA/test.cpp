#include "../MockCSVM.hpp"
#include "../compare.hpp"
#include "MockCUDA_CSVM.hpp"
#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#include "plssvm/backends/OpenMP/svm-kernel.hpp"
#include "plssvm/kernel_types.hpp"

#include <gtest/gtest.h>

#include <random>

TEST(IO, writeModel) {
    std::string model = std::tmpnam(nullptr);
    MockCUDA_CSVM csvm2(1., 0.001, plssvm::kernel_type::linear, 3.0, 0.0, 0.0, false);
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

    MockCUDA_CSVM csvm_CUDA;
    csvm_CUDA.libsvmParser(TESTFILE);
    csvm_CUDA.loadDataDevice();
    std::vector test = csvm_CUDA.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], test[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TEST(kernel, linear) {
    const size_t size = 512;
    std::vector<real_t> x1(size);
    std::vector<real_t> x2(size);
    std::generate(x1.begin(), x1.end(), std::rand);
    std::generate(x2.begin(), x2.end(), std::rand);
    real_t correct = linear_kernel(x1, x2);

    MockCUDA_CSVM csvm_CUDA;
    real_t result_CUDA = csvm_CUDA.kernel_function(x1, x2);
    real_t result2_CUDA = csvm_CUDA.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_CUDA);
    EXPECT_DOUBLE_EQ(correct, result2_CUDA);
}

TEST(learn, q_linear) {
    MockCSVM csvm;
    csvm.libsvmParser(TESTFILE);
    std::vector<real_t> correct = q<plssvm::kernel_type::linear>(csvm.get_data());

    MockCUDA_CSVM csvm_CUDA(1., 0.001, plssvm::kernel_type::linear);
    csvm_CUDA.libsvmParser(TESTFILE);
    csvm_CUDA.loadDataDevice();
    std::vector<real_t> test = csvm_CUDA.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], test[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

