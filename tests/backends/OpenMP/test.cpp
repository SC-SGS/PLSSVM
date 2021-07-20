/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Tests for the functionality related to the OpenMP backend.
 */

#include "MockOpenMP_CSVM.hpp"

#include "../../MockCSVM.hpp"  // MockCSVM
#include "../../utility.hpp"   // util::create_temp_file, util::gtest_expect_floating_point_eq

#include "../compare.hpp"
#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#include "plssvm/backends/OpenMP/svm-kernel.hpp"
#include "plssvm/kernel_types.hpp"

#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq

#include <cmath>       // std::abs
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>      // std::string

template <typename T>
class OpenMP : public ::testing::Test {};

using testing_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(OpenMP, testing_types);

TYPED_TEST(OpenMP, write_model) {
    // setup OpenMP C-SVM
    plssvm::parameter<TypeParam> params{ TESTPATH "/data/5x4.libsvm" };
    params.print_info = false;

    MockOpenMP_CSVM csvm{ params };

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

TYPED_TEST(OpenMP, q_linear) {
    // setup C-SVM
    plssvm::parameter<TypeParam> params{ TESTFILE };
    params.print_info = false;
    params.kernel = plssvm::kernel_type::linear;

    MockCSVM csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    // parse libsvm file and calculate q vector
    csvm.parse_libsvm(params.input_filename);
    const std::vector<real_type_csvm> correct = generate_q<plssvm::kernel_type::linear>(csvm.get_data());

    // setup OpenMP C-SVM
    MockOpenMP_CSVM csvm_openmp{ params };
    using real_type_csvm_openmp = typename decltype(csvm_openmp)::real_type;

    // check real_types
    ::testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_openmp>();

    // parse libsvm file and calculate q vector
    csvm_openmp.parse_libsvm(params.input_filename);
    csvm_openmp.setup_data_on_device();
    const std::vector<real_type_csvm_openmp> calculated = csvm_openmp.generate_q();

    // check size and values for correctness
    ASSERT_EQ(correct.size(), calculated.size());
    for (std::size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], calculated[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TYPED_TEST(OpenMP, kernel_linear) {
    // setup C-SVM
    plssvm::parameter<TypeParam> params{ TESTFILE };
    params.print_info = false;
    params.kernel = plssvm::kernel_type::linear;

    MockCSVM csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::real_type;

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
    const std::vector<real_type> q_vec = generate_q<plssvm::kernel_type::linear>(csvm.get_data());
    const real_type cost = csvm.get_cost();
    const real_type QA_cost = linear_kernel(csvm.get_data().back(), csvm.get_data().back()) + 1 / cost;

    for (const int add : { -1, 1 }) {
        const std::vector<real_type> correct = kernel_linear_function(csvm.get_data(), x, q_vec, QA_cost, cost, add);

        std::vector<real_type> calculated(dept, 0.0);
        plssvm::openmp::device_kernel_linear(q_vec, calculated, x, csvm.get_data(), QA_cost, real_type{ 1.0 } / cost, add);

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (std::size_t index = 0; index < correct.size(); ++index) {
            EXPECT_NEAR(correct[index], calculated[index], std::abs(correct[index] * 1e-10)) << " index: " << index << " add: " << add;
        }
    }
}

TYPED_TEST(OpenMP, q_polynomial) {
    // setup C-SVM
    plssvm::parameter<TypeParam> params{ TESTFILE };
    params.print_info = false;
    params.kernel = plssvm::kernel_type::polynomial;

    MockCSVM csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    // parse libsvm file and calculate q vector
    csvm.parse_libsvm(params.input_filename);
    const std::vector<real_type_csvm> correct = generate_q<plssvm::kernel_type::polynomial>(csvm.get_data(), csvm.get_degree(), csvm.get_gamma(), csvm.get_coef0());

    // setup OpenMP C-SVM
    MockOpenMP_CSVM csvm_openmp{ params };
    using real_type_csvm_openmp = typename decltype(csvm_openmp)::real_type;

    // check real_types
    ::testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_openmp>();

    // parse libsvm file and calculate q vector
    csvm_openmp.parse_libsvm(TESTFILE);
    csvm_openmp.setup_data_on_device();
    const std::vector<real_type_csvm_openmp> calculated = csvm_openmp.generate_q();

    // check size and values for correctness
    ASSERT_EQ(correct.size(), calculated.size());
    for (std::size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], calculated[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TYPED_TEST(OpenMP, kernel_polynomial) {
    // setup C-SVM
    plssvm::parameter<TypeParam> params{ TESTFILE };
    params.print_info = false;
    params.kernel = plssvm::kernel_type::polynomial;

    MockCSVM csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::real_type;

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
    const std::vector<real_type> q_vec = generate_q<plssvm::kernel_type::polynomial>(csvm.get_data(), csvm.get_degree(), csvm.get_gamma(), csvm.get_coef0());
    const real_type cost = csvm.get_cost();
    const real_type QA_cost = linear_kernel(csvm.get_data().back(), csvm.get_data().back()) + 1 / cost;

    for (const int add : { -1, 1 }) {
        const std::vector<real_type> correct = kernel_polynomial_function(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm.get_degree(), csvm.get_gamma(), csvm.get_coef0());

        std::vector<real_type> calculated(dept, 0.0);
        plssvm::openmp::device_kernel_poly(q_vec, calculated, x, csvm.get_data(), QA_cost, real_type{ 1.0 } / cost, add, csvm.get_degree(), csvm.get_gamma(), csvm.get_coef0());

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (std::size_t index = 0; index < correct.size(); ++index) {
            EXPECT_NEAR(correct[index], calculated[index], std::abs(correct[index] * 1e-10)) << " index: " << index << " add: " << add;
        }
    }
}

TYPED_TEST(OpenMP, q_radial) {
    // setup C-SVM
    plssvm::parameter<TypeParam> params{ TESTFILE };
    params.print_info = false;
    params.kernel = plssvm::kernel_type::rbf;

    MockCSVM csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    // parse libsvm file and calculate q vector
    csvm.parse_libsvm(params.input_filename);
    const std::vector<real_type_csvm> correct = generate_q<plssvm::kernel_type::rbf>(csvm.get_data(), csvm.get_gamma());

    // setup OpenMP C-SVM
    MockOpenMP_CSVM csvm_openmp{ params };
    using real_type_csvm_openmp = typename decltype(csvm_openmp)::real_type;

    // check real_types
    ::testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_openmp>();

    // parse libsvm file and calculate q vector
    csvm_openmp.parse_libsvm(params.input_filename);
    csvm_openmp.setup_data_on_device();
    const std::vector<real_type_csvm_openmp> calculated = csvm_openmp.generate_q();

    // check size and values for correctness
    ASSERT_EQ(correct.size(), calculated.size());
    for (std::size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], calculated[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TYPED_TEST(OpenMP, kernel_radial) {
    // setup C-SVM
    plssvm::parameter<TypeParam> params{ TESTFILE };
    params.print_info = false;
    params.kernel = plssvm::kernel_type::rbf;

    MockCSVM csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::real_type;

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
    const std::vector<real_type> q_vec = generate_q<plssvm::kernel_type::rbf>(csvm.get_data(), csvm.get_gamma());
    const real_type cost = csvm.get_cost();
    const real_type QA_cost = linear_kernel(csvm.get_data().back(), csvm.get_data().back()) + 1 / cost;

    for (const int add : { -1, 1 }) {
        const std::vector<real_type> correct = kernel_radial_function(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm.get_gamma());

        std::vector<real_type> calculated(dept, 0.0);
        plssvm::openmp::device_kernel_radial(q_vec, calculated, x, csvm.get_data(), QA_cost, real_type{ 1.0 } / cost, add, csvm.get_gamma());

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (std::size_t index = 0; index < correct.size(); ++index) {
            EXPECT_NEAR(correct[index], calculated[index], std::abs(correct[index] * 1e-10)) << " index: " << index << " add: " << add;
        }
    }
}
