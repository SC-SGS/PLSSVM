#include "../MockCSVM.hpp"
#include "../compare.hpp"
#include "MockCUDA_CSVM.hpp"
#include "plssvm/backends/CUDA/device_ptr.cuh"
#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#include "plssvm/backends/OpenMP/svm-kernel.hpp"
#include "plssvm/kernel_types.hpp"

#include <gtest/gtest.h>

#include <filesystem>
#include <fstream>
#include <random>

TEST(CUDA, writeModel) {
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

TEST(CUDA, q) {
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

TEST(CUDA, linear) {
    MockCUDA_CSVM csvm_CUDA(plssvm::kernel_type::linear);
    using real_type = typename MockCUDA_CSVM::real_type;

    const size_t size = 512;
    std::vector<real_type> x1(size);
    std::vector<real_type> x2(size);
    std::generate(x1.begin(), x1.end(), std::rand);
    std::generate(x2.begin(), x2.end(), std::rand);
    real_type correct = linear_kernel(x1, x2);

    real_type result_CUDA = csvm_CUDA.kernel_function(x1, x2);
    real_type result2_CUDA = csvm_CUDA.kernel_function(x1, x2);

    EXPECT_DOUBLE_EQ(correct, result_CUDA);
    EXPECT_DOUBLE_EQ(correct, result2_CUDA);
}

TEST(CUDA, q_linear) {
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

TEST(CUDA, kernel_linear) {
    MockCSVM csvm;
    using real_type = MockCSVM::real_type;

    csvm.parse_libsvm(TESTFILE);

    const size_t dept = csvm.get_num_data_points() - 1;

    std::vector<real_type> x(dept, 1.0);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(-1, 2.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

    const std::vector<real_type> q_ = q<plssvm::kernel_type::linear>(csvm.get_data());

    const real_type cost = csvm.cost_;

    const real_type QA_cost = linear_kernel(csvm.data_.back(), csvm.data_.back()) + 1 / cost;

    const size_t boundary_size = plssvm::THREADBLOCK_SIZE * plssvm::INTERNALBLOCK_SIZE;
    MockCUDA_CSVM csvm_CUDA(plssvm::kernel_type::linear);
    csvm_CUDA.parse_libsvm(TESTFILE);
    csvm_CUDA.setup_data_on_device();

    using namespace plssvm::detail;

    cuda::device_ptr<real_type> q_d{ dept + boundary_size };
    q_d.memcpy_to_device(q_, 0, dept);
    cuda::device_ptr<real_type> x_d{ dept + boundary_size };
    x_d.memcpy_to_device(x, 0, dept);
    cuda::device_ptr<real_type> r_d{ dept + boundary_size };
    r_d.memset(0);

    for (const real_type sgn : { -1.0, 1.0 }) {
        std::vector<real_type> correct = kernel_linear_function(csvm.get_data(), x, q_, sgn, QA_cost, cost);

        csvm_CUDA.QA_cost_ = QA_cost;
        csvm_CUDA.cost_ = cost;
        csvm_CUDA.run_device_kernel(0, q_d, r_d, x_d, csvm_CUDA.data_d_[0], static_cast<int>(sgn));

        cuda::device_synchronize();
        std::vector<real_type> result(dept, 0.0);
        r_d.memcpy_to_host(result, 0, dept);
        r_d.memset(0);

        ASSERT_EQ(correct.size(), result.size()) << "sgn: " << sgn;
        for (size_t index = 0; index < correct.size(); ++index) {
            EXPECT_NEAR(correct[index], result[index], std::abs(correct[index] * 1e-10)) << " index: " << index << " sgn: " << sgn;
        }
    }
}
