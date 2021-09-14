/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "mock_opencl_csvm.hpp"

#include "../../mock_csvm.hpp"  // mock_csvm
#include "../../utility.hpp"    // util::create_temp_file, util::gtest_expect_floating_point_eq, util::google_test::parameter_definition, util::google_test::parameter_definition_to_name
#include "../compare.hpp"       // compare::generate_q, compare::kernel_function, compare::device_kernel_function

#include "plssvm/backends/OpenCL/csvm.hpp"                  // plssvm::opencl::csvm
#include "plssvm/backends/OpenCL/detail/command_queue.hpp"  // plssvm::opencl::detail::command_queue
#include "plssvm/constants.hpp"                             // plssvm::THREAD_BLOCK_SIZE
#include "plssvm/detail/string_utility.hpp"                 // plssvm::detail::replace_all
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/parameter_train.hpp"                       // plssvm::parameter

#include <cmath>       // std::abs
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <iterator>    // std::istreambuf_iterator
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>      // std::string
#include <vector>      // std::vector

template <typename T>
class OpenCL_base : public ::testing::Test {};

// enumerate all type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

TYPED_TEST_SUITE(OpenCL_base, parameter_types);

TYPED_TEST(OpenCL_base, write_model) {
    // setup OpenCL C-SVM
    plssvm::parameter_train<typename TypeParam::real_type> params{ TEST_PATH "/data/5x4.libsvm" };
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    mock_opencl_csvm csvm{ params };

    // create temporary model file
    std::string model_file = util::create_temp_file();
    // learn model
    csvm.learn();
    // write learned model to file
    csvm.write_model(model_file);

    // read content of model file and delete it
    std::ifstream model_ifs(model_file);
    std::string file_content((std::istreambuf_iterator<char>(model_ifs)), std::istreambuf_iterator<char>());
    std::filesystem::remove(model_file);

    // check model file content for correctness
    switch (params.kernel) {
        case plssvm::kernel_type::linear:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
            break;
        case plssvm::kernel_type::polynomial:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type polynomial\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
            break;
        case plssvm::kernel_type::rbf:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type rbf\ndegree [0-9]+\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\ncoef0 [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
            break;
        default:
            FAIL() << "untested kernel" << params.kernel;
            break;
    }
}

// generate tests for the generation of the q vector
template <typename T>
class OpenCL_generate_q : public ::testing::Test {};
TYPED_TEST_SUITE(OpenCL_generate_q, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(OpenCL_generate_q, generate_q) {
    // setup C-SVM
    plssvm::parameter_train<typename TypeParam::real_type> params{ TEST_FILE };
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    mock_csvm csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    // parse libsvm file and calculate q vector
    const std::vector<real_type_csvm> correct = compare::generate_q<TypeParam::kernel>(csvm.get_data(), csvm);

    // setup OpenCL C-SVM
    mock_opencl_csvm csvm_opencl{ params };
    using real_type_csvm_opencl = typename decltype(csvm_opencl)::real_type;

    // check real_types
    ::testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_opencl>();

    // parse libsvm file and calculate q vector
    csvm_opencl.setup_data_on_device();
    const std::vector<real_type_csvm_opencl> calculated = csvm_opencl.generate_q();

    ASSERT_EQ(correct.size(), calculated.size());
    for (std::size_t index = 0; index < correct.size(); ++index) {
        util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}", index));
    }
}

// generate tests for the device kernel functions
template <typename T>
class OpenCL_device_kernel : public ::testing::Test {};
TYPED_TEST_SUITE(OpenCL_device_kernel, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(OpenCL_device_kernel, device_kernel) {
    // setup C-SVM
    plssvm::parameter_train<typename TypeParam::real_type> params{ TEST_FILE };
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    const size_type dept = csvm.get_num_data_points() - 1;
    // const size_type num_features = csvm.get_num_features();

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

    // setup OpenCL C-SVM
    mock_opencl_csvm csvm_opencl{ params };

    // setup data on device
    csvm_opencl.setup_data_on_device();

    // TODO: multi GPU support
    plssvm::opencl::detail::command_queue &queue = csvm_opencl.get_devices()[0];
    const size_type boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    plssvm::opencl::detail::device_ptr<real_type> q_d{ dept + boundary_size, queue };
    q_d.memcpy_to_device(q_vec, 0, dept);
    plssvm::opencl::detail::device_ptr<real_type> x_d{ dept + boundary_size, queue };
    x_d.memcpy_to_device(x, 0, dept);
    plssvm::opencl::detail::device_ptr<real_type> r_d{ dept + boundary_size, queue };
    r_d.memset(0);

    for (const int add : { -1, 1 }) {
        std::vector<real_type> correct = compare::device_kernel_function<TypeParam::kernel>(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm);

        csvm_opencl.set_QA_cost(QA_cost);
        csvm_opencl.set_cost(cost);

        csvm_opencl.run_device_kernel(0, q_d, r_d, x_d, csvm_opencl.get_device_data()[0], add);

        std::vector<real_type> calculated(dept);
        r_d.memcpy_to_host(calculated, 0, dept);
        r_d.memset(0);

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (size_t index = 0; index < correct.size(); ++index) {
            util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}, add: {}", index, add));
        }
    }
}

// enumerate double and kernel combinations to test
using parameter_types_double = ::testing::Types<
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

template <typename T>
class OpenCL_accuracy : public ::testing::Test {};
TYPED_TEST_SUITE(OpenCL_accuracy, parameter_types_double, util::google_test::parameter_definition_to_name);  // TODO: float parameter_types accuracy
TYPED_TEST(OpenCL_accuracy, accuracy) {
    plssvm::parameter_train<typename TypeParam::real_type> params{ TEST_FILE };
    params.print_info = false;
    params.kernel = TypeParam::kernel;
    params.epsilon = 0.0000000001;

    // setup OpenCL C-SVM
    mock_opencl_csvm csvm_opencl{ params };
    using real_type_csvm_opencl = typename decltype(csvm_opencl)::real_type;

    // learn
    csvm_opencl.learn();

    real_type_csvm_opencl acc = csvm_opencl.accuracy();
    ASSERT_GT(acc, 0.95);
}
