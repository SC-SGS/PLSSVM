/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Tests for the functionality related to the OpenMP backend.
 */

#include "MockCUDA_CSVM.hpp"

#include "../../MockCSVM.hpp"  // MockCSVM
#include "../../utility.hpp"   // util::create_temp_file, util::gtest_expect_floating_point_eq

#include "../compare.hpp"                      // compare::generate_q, compare::kernel_function, compare::device_kernel_function
#include "plssvm/backends/CUDA/CUDA_CSVM.hpp"  // plssvm::OpenMP_CSVM
#include "plssvm/kernel_types.hpp"             // plssvm::kernel_type
#include "plssvm/parameter.hpp"                // plssvm::parameter
#include "plssvm/typedef.hpp"                  // plssvm::THREAD_BLOCK_SIZE

#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq, ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ, EXPECT_EQ, EXPECT_THAT, EXPECT_THROW

#include <cmath>       // std::abs
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <iterator>    // std::istreambuf_iterator
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>      // std::string
#include <vector>      // std::vector

template <typename T>
class CUDA : public ::testing::Test {};

using testing_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CUDA, testing_types);

TYPED_TEST(CUDA, write_model) {
    // setup CUDA C-SVM
    plssvm::parameter<TypeParam> params{ TESTPATH "/data/5x4.libsvm" };
    params.print_info = false;

    MockCUDA_CSVM csvm{ params };

    // create temporary model file
    std::string model_file = util::create_temp_file();

    // learn
    csvm.learn(params.input_filename, model_file);

    // read content of model file and delete it
    std::ifstream model_ifs(model_file);
    std::string file_content((std::istreambuf_iterator<char>(model_ifs)), std::istreambuf_iterator<char>());
    std::filesystem::remove(model_file);

    // check model file content for correctness
    EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv [1-9][0-9]*\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
}

TYPED_TEST(CUDA, q_linear) {
    // setup C-SVM
    plssvm::parameter<TypeParam> params{ TESTFILE };
    params.print_info = false;
    params.kernel = plssvm::kernel_type::linear;

    MockCSVM csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    // parse libsvm file and calculate q vector
    csvm.parse_libsvm(params.input_filename);
    const std::vector<real_type_csvm> correct = compare::generate_q<plssvm::kernel_type::linear>(csvm.get_data());

    // setup CUDA C-SVM
    MockCUDA_CSVM csvm_cuda{ params };
    using real_type_csvm_cuda = typename decltype(csvm_cuda)::real_type;

    // check real_types
    ::testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_cuda>();

    // parse libsvm file and calculate q vector
    csvm_cuda.parse_libsvm(params.input_filename);
    csvm_cuda.setup_data_on_device();
    const std::vector<real_type_csvm_cuda> calculated = csvm_cuda.generate_q();

    ASSERT_EQ(correct.size(), calculated.size());
    for (std::size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], calculated[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TYPED_TEST(CUDA, kernel_linear) {
    // setup C-SVM
    plssvm::parameter<TypeParam> params{ TESTFILE };
    params.print_info = false;
    params.kernel = plssvm::kernel_type::linear;

    MockCSVM csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    // parse libsvm file
    csvm.parse_libsvm(params.input_filename);

    const size_type dept = csvm.get_num_data_points() - 1;

    // create x vector and fill it with random values
    std::vector<real_type> x(dept);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(-1, 2.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

    // create correct q vector, cost and QA_cost
    const std::vector<real_type> q_vec = compare::generate_q<plssvm::kernel_type::linear>(csvm.get_data());
    const real_type cost = csvm.get_cost();
    const real_type QA_cost = compare::kernel_function<plssvm::kernel_type::linear>(csvm.get_data().back(), csvm.get_data().back()) + 1 / cost;

    // setup CUDA C-SVM
    MockCUDA_CSVM csvm_cuda{ params };

    // parse libsvm file
    csvm_cuda.parse_libsvm(params.input_filename);

    // setup data on device
    csvm_cuda.setup_data_on_device();

    const size_type boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    plssvm::cuda::detail::device_ptr<real_type> q_d{ dept + boundary_size };
    q_d.memcpy_to_device(q_vec, 0, dept);
    plssvm::cuda::detail::device_ptr<real_type> x_d{ dept + boundary_size };
    x_d.memcpy_to_device(x, 0, dept);
    plssvm::cuda::detail::device_ptr<real_type> r_d{ dept + boundary_size };
    r_d.memset(0);

    for (const int add : { -1, 1 }) {
        const std::vector<real_type> correct = compare::device_kernel_function<plssvm::kernel_type::linear>(csvm.get_data(), x, q_vec, QA_cost, cost, add);

        csvm_cuda.QA_cost_ = QA_cost;
        csvm_cuda.cost_ = cost;
        csvm_cuda.run_device_kernel(0, q_d, r_d, x_d, csvm_cuda.data_d_[0], add);

        plssvm::cuda::detail::device_synchronize();
        std::vector<real_type> calculated(dept);
        r_d.memcpy_to_host(calculated, 0, dept);
        r_d.memset(0);

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (std::size_t index = 0; index < correct.size(); ++index) {
            EXPECT_NEAR(correct[index], calculated[index], std::abs(correct[index] * 1e-10)) << " index: " << index << " add: " << add;
        }
    }
}
