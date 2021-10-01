/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Tests for the functionality related to the SYCL backend.
 */

#include "mock_sycl_csvm.hpp"

#include "../../mock_csvm.hpp"  // mock_csvm
#include "../../utility.hpp"    // util::create_temp_file, util::gtest_expect_correct_csvm_factory
#include "../compare.hpp"       // compare::generate_q, compare::kernel_function, compare::device_kernel_function

#include "plssvm/backends/SYCL/csvm.hpp"               // plssvm::sycl::csvm
#include "plssvm/backends/SYCL/detail/device_ptr.hpp"  // plssvm::sycl::detail::device_ptr
#include "plssvm/constants.hpp"                        // plssvm::THREAD_BLOCK_SIZE
#include "plssvm/detail/string_conversion.hpp"         // plssvm::detail::convert_to
#include "plssvm/detail/string_utility.hpp"            // plssvm::detail::replace_all
#include "plssvm/kernel_types.hpp"                     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter

#include "sycl/sycl.hpp"  // sycl::queue
#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq, ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ, EXPECT_EQ, EXPECT_THAT, EXPECT_THROW_WHAT

#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <iterator>    // std::istreambuf_iterator
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>      // std::string
#include <vector>      // std::vector

// enumerate all floating point type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

template <typename T>
class SYCL_CSVM : public ::testing::Test {};
TYPED_TEST_SUITE(SYCL_CSVM, parameter_types, util::google_test::parameter_definition_to_name);

// check whether the csvm factory function correctly creates a sycl::csvm
TYPED_TEST(SYCL_CSVM, csvm_factory) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.backend = plssvm::backend_type::sycl;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    util::gtest_expect_correct_csvm_factory<plssvm::sycl::csvm>(params);
}

// check whether the q vector is generated correctly
TYPED_TEST(SYCL_CSVM, generate_q) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_FILE);

    // create base C-SVM
    mock_csvm csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    // calculate q vector
    const std::vector<real_type_csvm> correct = compare::generate_q<TypeParam::kernel>(csvm.get_data(), csvm);

    // create C-SVM using the SYCL backend
    mock_sycl_csvm csvm_sycl{ params };
    using real_type_csvm_sycl = typename decltype(csvm_sycl)::real_type;

    // check real_types
    ::testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_sycl>();

    // parse libsvm file and calculate q vector
    csvm_sycl.setup_data_on_device();
    const std::vector<real_type_csvm_sycl> calculated = csvm_sycl.generate_q();

    ASSERT_EQ(correct.size(), calculated.size());
    for (std::size_t index = 0; index < correct.size(); ++index) {
        util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}", index));
    }
}

// check whether the device kernels are correct
TYPED_TEST(SYCL_CSVM, device_kernel) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_FILE);

    // create base C-SVM
    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    const size_type dept = csvm.get_num_data_points() - 1;

    // create x vector and fill it with random values
    std::vector<real_type> x(dept);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(1.0, 2.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

    // create correct q vector, cost and QA_cost
    const std::vector<real_type> q_vec = compare::generate_q<TypeParam::kernel>(csvm.get_data(), csvm);
    const real_type cost = csvm.get_cost();
    const real_type QA_cost = compare::kernel_function<TypeParam::kernel>(csvm.get_data().back(), csvm.get_data().back(), csvm) + 1 / cost;

    // create C-SVM using the SYCL backend
    mock_sycl_csvm csvm_sycl{ params };

    // setup data on device
    csvm_sycl.setup_data_on_device();

    // TODO: multi GPU support
    // setup all additional data
    sycl::queue &q = csvm_sycl.get_devices()[0];
    const size_type boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    plssvm::sycl::detail::device_ptr<real_type> q_d{ dept + boundary_size, q };
    q_d.memcpy_to_device(q_vec, 0, dept);
    plssvm::sycl::detail::device_ptr<real_type> x_d{ dept + boundary_size, q };
    x_d.memcpy_to_device(x, 0, dept);
    plssvm::sycl::detail::device_ptr<real_type> r_d{ dept + boundary_size, q };
    r_d.memset(0);

    for (const auto add : { real_type{ -1 }, real_type{ 1 } }) {
        const std::vector<real_type> correct = compare::device_kernel_function<TypeParam::kernel>(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm);

        csvm_sycl.set_QA_cost(QA_cost);
        csvm_sycl.set_cost(cost);
        plssvm::sycl::detail::device_synchronize(q);
        csvm_sycl.run_device_kernel(0, q_d, r_d, x_d, csvm_sycl.get_device_data()[0], add);

        plssvm::sycl::detail::device_synchronize(q);
        std::vector<real_type> calculated(dept);
        r_d.memcpy_to_host(calculated, 0, dept);
        r_d.memset(0);

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (std::size_t index = 0; index < correct.size(); ++index) {
            util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}, add: {}", index, add));
        }
    }
}

// check whether the correct labels are predicted
TYPED_TEST(SYCL_CSVM, predict) {
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;

    params.parse_model_file(TEST_PATH "/data/models/500x200.libsvm." + fmt::format("{}", TypeParam::kernel) + ".model");
    params.parse_test_file(TEST_PATH "/data/libsvm/500x200.libsvm.test");

    // setup SYCL C-SVM
    mock_sycl_csvm csvm_sycl{ params };
    using real_type = typename decltype(csvm_sycl)::real_type;
    using size_type = typename decltype(csvm_sycl)::size_type;

    // predict
    std::vector<real_type> predicted_values = csvm_sycl.predict_label(*params.test_data_ptr);
    std::vector<real_type> predicted_values_real = csvm_sycl.predict(*params.test_data_ptr);

    // read correct prediction
    std::ifstream ifs(TEST_PATH "/data/predict/500x200.libsvm.predict");
    std::string line;
    std::vector<real_type> correct_values;
    correct_values.reserve(500);
    while (std::getline(ifs, line, '\n')) {
        correct_values.push_back(plssvm::detail::convert_to<real_type>(line));
    }

    ASSERT_EQ(correct_values.size(), predicted_values.size());
    for (size_type i = 0; i < correct_values.size(); ++i) {
        EXPECT_EQ(correct_values[i], predicted_values[i]) << "data point: " << i << " real value: " << predicted_values_real[i];
        if (correct_values[i] > real_type{ 0 }) {
            EXPECT_GT(predicted_values_real[i], real_type{ 0 });
        } else {
            EXPECT_LT(predicted_values_real[i], real_type{ 0 });
        }
    }
}

// check whether the accuracy calculation is correct
TYPED_TEST(SYCL_CSVM, accuracy) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_FILE);

    // create C-SVM using the SYCL backend
    mock_sycl_csvm csvm_sycl{ params };
    using real_type = typename decltype(csvm_sycl)::real_type;
    using size_type = typename decltype(csvm_sycl)::size_type;

    // learn
    csvm_sycl.learn();

    // predict label and calculate correct accuracy
    std::vector<real_type> label_predicted = csvm_sycl.predict_label(*params.data_ptr);
    ASSERT_EQ(label_predicted.size(), params.value_ptr->size());
    size_type count = 0;
    for (size_type i = 0; i < label_predicted.size(); ++i) {
        if (label_predicted[i] == (*params.value_ptr)[i]) {
            ++count;
        }
    }
    real_type accuracy_correct = static_cast<real_type>(count) / static_cast<real_type>(label_predicted.size());

    // calculate accuracy
    real_type accuracy_calculated = csvm_sycl.accuracy();
    util::gtest_assert_floating_point_eq(accuracy_calculated, accuracy_correct);
}