#include "../MockCSVM.hpp"
#include "../compare.hpp"
#include "MockCUDA_CSVM.hpp"
#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#include "plssvm/backends/OpenMP/svm-kernel.hpp"
#include "plssvm/kernel_types.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>

TEST(IO, writeModel) {
    std::string model = std::filesystem::temp_directory_path().string();
    model += "/tmpfile_XXXXXX";
    // create unique temporary file
    int fd = mkstemp(model.data());
    // immediately close file if possible
    if (fd >= 0) {
        close(fd);
    }

    MockCUDA_CSVM csvm2(plssvm::kernel_type::linear, 3.0, 0.0, 0.0, 1., 0.001, false);
    std::string testfile = TESTPATH "/data/5x4.libsvm";
    csvm2.learn(testfile, model);

    std::ifstream model_ifs(model);
    std::string genfile2((std::istreambuf_iterator<char>(model_ifs)), std::istreambuf_iterator<char>());
    std::filesystem::remove(model);

    EXPECT_THAT(genfile2, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv [1-9][0-9]*\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
}

TEST(learn, q) {
    MockCUDA_CSVM csvm_CUDA;
    using real_type = typename MockCUDA_CSVM::real_type;

    std::vector correct = generate_q<real_type>(TESTFILE);

    csvm_CUDA.parse_libsvm(TESTFILE);
    csvm_CUDA.setup_data_on_device();
    std::vector<real_type> test = csvm_CUDA.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], test[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TEST(kernel, linear) {
    MockCUDA_CSVM csvm_CUDA(plssvm::kernel_type::linear);
    using real_type = typename MockCUDA_CSVM::real_type;

    const size_t size = 512;
    std::vector<real_type> x1(size);
    std::vector<real_type> x2(size);
    std::generate(x1.begin(), x1.end(), std::rand);
    std::generate(x2.begin(), x2.end(), std::rand);
    real_type correct = linear_kernel(x1, x2);

    real_type result_CUDA = csvm_CUDA.kernel_function(x1, x2);
    real_type result2_CUDA = csvm_CUDA.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_CUDA);
    EXPECT_DOUBLE_EQ(correct, result2_CUDA);
}

TEST(learn, q_linear) {
    MockCSVM csvm;
    using real_type = typename MockCSVM::real_type;

    csvm.parse_libsvm(TESTFILE);
    std::vector<real_type> correct = q<plssvm::kernel_type::linear>(csvm.get_data());

    MockCUDA_CSVM csvm_CUDA(plssvm::kernel_type::linear);
    csvm_CUDA.parse_libsvm(TESTFILE);
    csvm_CUDA.setup_data_on_device();
    std::vector<real_type> test = csvm_CUDA.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], test[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}
