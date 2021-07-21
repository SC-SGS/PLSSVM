/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Tests for the functionality related to the OpenMP backend.
 */

#include "MockOpenMP_CSVM.hpp"

#include "../../MockCSVM.hpp"  // MockCSVM
#include "../../utility.hpp"   // util::create_temp_file, util::gtest_expect_floating_point_eq, util::google_test::parameter_definition, util::google_test::parameter_definition_to_name

#include "../compare.hpp"                          // compare::generate_q, compare::kernel_function, compare::device_kernel_function
#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"  // plssvm::OpenMP_CSVM
#include "plssvm/backends/OpenMP/svm-kernel.hpp"   // plssvm::openmp::device_kernel_linear, plssvm::openmp::device_kernel_poly, plssvm::openmp::device_kernel_radial
#include "plssvm/kernel_types.hpp"                 // plssvm::kernel_type
#include "plssvm/parameter.hpp"                    // plssvm::parameter

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
class OpenMP_base : public ::testing::Test {};

using write_model_parameter_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(OpenMP_base, write_model_parameter_types);

TYPED_TEST(OpenMP_base, write_model) {
    // setup OpenMP C-SVM
    plssvm::parameter<TypeParam> params{ TEST_PATH "/data/5x4.libsvm" };
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

// enumerate all type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

// generate tests for the generation of the q vector
template <typename T>
class OpenMP_generate_q : public ::testing::Test {};
TYPED_TEST_SUITE(OpenMP_generate_q, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(OpenMP_generate_q, generate_q) {
    // setup C-SVM
    plssvm::parameter<typename TypeParam::real_type> params{ TEST_FILE };
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    MockCSVM csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    // parse libsvm file and calculate q vector
    csvm.parse_libsvm(params.input_filename);
    const std::vector<real_type_csvm> correct = compare::generate_q<TypeParam::kernel>(csvm.get_data(), csvm);

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

// generate tests for the device kernel functions
template <typename T>
class OpenMP_device_kernel : public ::testing::Test {};
TYPED_TEST_SUITE(OpenMP_device_kernel, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(OpenMP_device_kernel, device_kernel) {
    // setup C-SVM
    plssvm::parameter<typename TypeParam::real_type> params{ TEST_FILE };
    params.print_info = false;
    params.kernel = TypeParam::kernel;

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
    const std::vector<real_type> q_vec = compare::generate_q<TypeParam::kernel>(csvm.get_data(), csvm);
    const real_type cost = csvm.get_cost();
    const real_type QA_cost = compare::kernel_function<TypeParam::kernel>(csvm.get_data().back(), csvm.get_data().back(), csvm) + 1 / cost;

    // setup OpenMP C-SVM
    MockOpenMP_CSVM csvm_openmp{ params };

    // parse libsvm file
    csvm_openmp.parse_libsvm(params.input_filename);

    // setup data on device
    csvm_openmp.setup_data_on_device();

    for (const int add : { -1, 1 }) {
        const std::vector<real_type> correct = compare::device_kernel_function<TypeParam::kernel>(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm);

        std::vector<real_type> calculated(dept, 0.0);
        csvm_openmp.set_QA_cost(QA_cost);
        csvm_openmp.set_cost(cost);
        csvm_openmp.run_device_kernel(q_vec, calculated, x, csvm_openmp.get_device_data(), add);

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (std::size_t index = 0; index < correct.size(); ++index) {
            EXPECT_NEAR(correct[index], calculated[index], std::abs(correct[index] * 1e-10)) << " index: " << index << " add: " << add;
        }
    }
}
