/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic tests for all backends to reduce code duplication.
 */

#pragma once

#include "backends/compare.hpp"  // compare::generate_q, compare::kernel_function, compare::device_kernel_function
#include "mock_csvm.hpp"         // mock_csvm
#include "utility.hpp"           // util::gtest_assert_floating_point_near, util::gtest_assert_floating_point_eq, util::gtest_expect_correct_csvm_factory, util::create_temp_file

#include "plssvm/backend_types.hpp"                         // plssvm::backend_type
#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type
#include "plssvm/constants.hpp"                             // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/csvm_factory.hpp"
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::exception
#include "plssvm/kernel_function_types.hpp"     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                 // plssvm::parameter

#include "custom_test_macros.hpp"

#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads
#include "gmock/gmock.h"  // EXPECT_THAT
#include "gtest/gtest.h"  // ASSERT_GT, ASSERT_TRUE, ASSERT_EQ, EXPECT_EQ, EXPECT_GT, testing::ContainsRegex, testing::StaticAssertTypeEq

#include <algorithm>   // std::generate
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <regex>       // std::regex, std::regex_match
#include <string>      // std::string, std::getline
#include <vector>      // std::vector

namespace generic {

template <typename real_type, typename mock_csvm_type>
inline void test_solve_system_of_linear_equations(const plssvm::kernel_function_type kernel_type) {
    // create parameter struct
    plssvm::detail::parameter<real_type> params{};
    params.kernel_type = kernel_type;
    params.cost = 2.0;

    // create the data that should be used
    // Matrix with 1-1/cost on main diagonal. Thus, the diagonal entries become one with the additional addition of 1/cost
    const std::vector<std::vector<real_type>> data = {
        { real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) } }
    };
    const std::vector<real_type> rhs{ real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 }, real_type{ 4.0 }, real_type{ 5.0 } };

    // create C-SVM: must be done using the mock class, since solve_system_of_linear_equations_impl is protected
    const mock_csvm_type svm{};

    // solve the system of linear equations using the CG algorithm: (A + (I * 1 / cost))x =b  => Ix = b => should be trivial x = b
    const std::pair<std::vector<real_type>, real_type> calculated = svm.solve_system_of_linear_equations_impl(params, data, rhs, real_type{ 0.00001 }, 1);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(rhs, calculated.first);
    // TODO: test rho
}

template <typename real_type, typename mock_csvm_type>
inline void test_predict_values(const plssvm::kernel_function_type kernel_type) {
    // create parameter struct
    plssvm::detail::parameter<real_type> params{};
    params.kernel_type = kernel_type;

    // create the data that should be used
    // TODO: meaningful data
    const std::vector<std::vector<real_type>> support_vectors{ { real_type{ 1.0 } } };
    const std::vector<real_type> weights{ real_type{ 0.5 } };
    const real_type rho{};
    std::vector<real_type> w{};
    const std::vector<std::vector<real_type>> data{ { real_type{ 2.0 } } };

    // create C-SVM: must be done using the mock class, since solve_system_of_linear_equations_impl is protected
    const mock_csvm_type svm{};

    // predict the values using the previously learned support vectors and weights
    const std::vector<real_type> calculated = svm.predict_values_impl(params, support_vectors, weights, rho, w, data);

    // check the calculated result for correctness
    ASSERT_EQ(calculated.size(), data.size());
    // TODO: add other tests
    // in case of the linear kernel, the w vector should have been filled
    if (kernel_type == plssvm::kernel_function_type::linear) {
        EXPECT_EQ(w.size(), support_vectors.front().size());
    } else {
        EXPECT_TRUE(w.empty());
    }
}

template <typename real_type, typename csvm_type>
inline void test_predict(const plssvm::kernel_function_type kernel_type) {
    // create parameter struct
    plssvm::parameter params{};
    params.kernel_type = kernel_type;

    // create data set to be used
    const plssvm::data_set<real_type> test_data{ PLSSVM_TEST_PATH "/data/predict/500x200_test.libsvm" };

    // read the previously learned model
    const plssvm::model<real_type> model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/500x200_{}.libsvm.model", kernel_type) };

    // create C-SVM
    const csvm_type svm{ params };

    // predict label
    const std::vector<int> calculated = svm.predict(model, test_data);

    // read ground truth from file
    std::ifstream prediction_file{ PLSSVM_TEST_PATH "/data/predict/500x200.libsvm.predict" };
    const std::vector<int> ground_truth{ std::istream_iterator<int>{ prediction_file }, std::istream_iterator<int>{} };

    // check the calculated result for correctness
    EXPECT_EQ(calculated, ground_truth);
}

template <typename real_type, typename csvm_type>
inline void test_score(const plssvm::kernel_function_type kernel_type) {
    // create parameter struct
    plssvm::parameter params{};
    params.kernel_type = kernel_type;

    // create data set to be used
    const plssvm::data_set<real_type> test_data{ PLSSVM_TEST_PATH "/data/predict/500x200_test.libsvm" };

    // read the previously learned model
    const plssvm::model<real_type> model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/500x200_{}.libsvm.model", kernel_type) };

    // create C-SVM
    const plssvm::openmp::csvm svm{ params };

    // predict label
    const real_type calculated = svm.score(model, test_data);

    // check the calculated result for correctness
    EXPECT_EQ(calculated, real_type{ 1.0 });
}

// template <template <typename> typename csvm_type, typename real_type, plssvm::kernel_type kernel>
// inline void generate_q_test() {
//     // create parameter object
//     plssvm::parameter<real_type> params;
//     params.print_info = false;
//     params.kernel = kernel;
//
//     params.parse_train_file(PLSSVM_TEST_FILE);
//
//     // create base C-SVM
//     mock_csvm csvm{ params };
//     using real_type_csvm = typename decltype(csvm)::real_type;
//
//     // setup C-SVM based on specified backend
//     csvm_type csvm_backend{ params };
//     using real_type_csvm_backend = typename decltype(csvm_backend)::real_type;
//
//     // calculate q vector
//     const std::vector<real_type> correct = compare::generate_q<kernel>(csvm.get_data(), csvm_backend.get_num_devices(), csvm);
//
//     // check types
//     testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_backend>();
//
//     // calculate q vector
//     csvm_backend.setup_data_on_device();
//     const std::vector<real_type> calculated = csvm_backend.generate_q();
//
//     ASSERT_EQ(correct.size(), calculated.size());
//     for (typename std::vector<real_type>::size_type index = 0; index < correct.size(); ++index) {
//         util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}", index));
//     }
// }
//
// template <template <typename> typename csvm_type, typename real_type, plssvm::kernel_type kernel, plssvm::sycl_generic::kernel_invocation_type invocation_type = plssvm::sycl_generic::kernel_invocation_type::automatic>
// inline void device_kernel_test() {
//     // create parameter object
//     plssvm::parameter<real_type> params;
//     params.print_info = false;
//     params.kernel = kernel;
//     params.sycl_kernel_invocation_type = invocation_type;
//
//     params.parse_train_file(PLSSVM_TEST_FILE);
//
//     // create base C-SVM
//     mock_csvm csvm{ params };
//     using real_type_csvm = typename decltype(csvm)::real_type;
//
//     const std::size_t dept = csvm.get_num_data_points() - 1;
//
//     // create x vector and fill it with random values
//     std::vector<real_type> x(dept);
//     std::random_device rd;
//     std::mt19937 gen(rd());
//     std::uniform_real_distribution<real_type> dist(1.0, 2.0);
//     std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
//
//     // create C-SVM using the specified backend
//     csvm_type csvm_backend{ params };
//     using real_type_csvm_backend = typename decltype(csvm)::real_type;
//     using device_ptr_type = typename decltype(csvm_backend)::device_ptr_type;
//     using queue_type = typename decltype(csvm_backend)::queue_type;
//
//     // create correct q vector, cost and QA_cost
//     const std::vector<real_type> q_vec = compare::generate_q<kernel>(csvm.get_data(), csvm_backend.get_num_devices(), csvm);
//     const real_type cost = csvm.get_cost();
//     const real_type QA_cost = compare::kernel_function<kernel>(csvm.get_data().back(), csvm.get_data().back(), csvm) + 1 / cost;
//
//     // check types
//     testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_backend>();
//
//     // setup data on device
//     csvm_backend.setup_data_on_device();
//
//     // setup all additional data
//     const std::size_t boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
//     std::vector<device_ptr_type> q_d{};
//     std::vector<device_ptr_type> x_d{};
//     std::vector<device_ptr_type> r_d{};
//
//     for (queue_type &queue : csvm_backend.get_devices()) {
//         q_d.emplace_back(dept + boundary_size, queue).memcpy_to_device(q_vec, 0, dept);
//         x_d.emplace_back(dept + boundary_size, queue).memset(0);
//         x_d.back().memcpy_to_device(x, 0, dept);
//         r_d.emplace_back(dept + boundary_size, queue).memset(0);
//     }
//
//     for (const auto add : { real_type{ -1 }, real_type{ 1 } }) {
//         const std::vector<real_type> correct = compare::device_kernel_function<kernel>(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm);
//
//         csvm_backend.set_QA_cost(QA_cost);
//         csvm_backend.set_cost(cost);
//
//         for (typename std::vector<queue_type>::size_type device = 0; device < csvm_backend.get_num_devices(); ++device) {
//             csvm_backend.run_device_kernel(device, q_d[device], r_d[device], x_d[device], add);
//         }
//         std::vector<real_type> calculated(dept);
//         csvm_backend.device_reduction(r_d, calculated);
//
//         for (device_ptr_type &r_d_device : r_d) {
//             r_d_device.memset(0);
//         }
//
//         ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
//         for (typename std::vector<real_type>::size_type index = 0; index < correct.size(); ++index) {
//             util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}, add: {}", index, add));
//         }
//     }
// }
//
// template <template <typename> typename csvm_type, typename real_type, plssvm::kernel_type kernel>
// inline void predict_test() {
//     // create parameter object
//     plssvm::parameter<real_type> params;
//     params.print_info = false;
//
//     params.parse_model_file(fmt::format(PLSSVM_TEST_PATH "/data/models/500x200.libsvm.{}.model", kernel));
//     params.parse_test_file(PLSSVM_TEST_PATH "/data/libsvm/500x200.libsvm.test");
//
//     // create C-SVM using the specified backend
//     csvm_type csvm{ params };
//
//     // predict
//     std::vector<real_type> predicted_values = csvm.predict_label(*params.test_data_ptr);
//     std::vector<real_type> predicted_values_real = csvm.predict(*params.test_data_ptr);
//
//     // read correct prediction
//     std::ifstream ifs(PLSSVM_TEST_PATH "/data/predict/500x200.libsvm.predict");
//     std::string line;
//     std::vector<real_type> correct_values;
//     correct_values.reserve(500);
//     while (std::getline(ifs, line, '\n')) {
//         correct_values.push_back(plssvm::detail::convert_to<real_type>(line));
//     }
//
//     ASSERT_EQ(correct_values.size(), predicted_values.size());
//     for (typename std::vector<real_type>::size_type i = 0; i < correct_values.size(); ++i) {
//         EXPECT_EQ(correct_values[i], predicted_values[i]) << "data point: " << i << " real value: " << predicted_values_real[i];
//         EXPECT_GT(correct_values[i] * predicted_values_real[i], real_type{ 0 });
//     }
//
//     // empty points should return an empty vector
//     std::vector<std::vector<real_type>> points;
//     EXPECT_EQ(std::vector<real_type>{}, csvm.predict(points));
//
//     // test exceptions
//     [[maybe_unused]] std::vector<real_type> acc_vec;
//     // the number of features of the prediction points must all be the same
//     points = { { real_type{ 1 }, real_type{ 2 } }, { real_type{ 3 } } };
//     EXPECT_THROW_WHAT(acc_vec = csvm.predict(points), plssvm::exception, "All points in the prediction point vector must have the same number of features!");
//
//     // the number of features of the prediction points must match the number of features of the data points
//     points = { { real_type{ 1 }, real_type{ 2 } }, { real_type{ 3 }, real_type{ 4 } } };
//     EXPECT_THROW_WHAT(acc_vec = csvm.predict_label(points), plssvm::exception, fmt::format("Number of features per data point ({}) must match the number of features per predict point (2)!", params.data_ptr->front().size()));
//
//     // alpha values must not be nullptr in order to predict
//     csvm.get_alpha_ptr() = nullptr;
//     EXPECT_THROW_WHAT(acc_vec = csvm.predict_label(*params.test_data_ptr), plssvm::exception, "No alphas provided for prediction!");
// }
//
// template <template <typename> typename csvm_type, typename real_type, plssvm::kernel_type kernel>
// inline void accuracy_test() {
//     // create parameter object
//     plssvm::parameter<real_type> params;
//     params.print_info = false;
//     params.kernel = kernel;
//
//     params.parse_train_file(PLSSVM_TEST_FILE);
//
//     // create C-SVM using the specified backend
//     csvm_type csvm{ params };
//
//     // learn
//     csvm.learn();
//
//     // predict label and calculate correct accuracy
//     const std::vector<real_type> label_predicted = csvm.predict_label(*params.data_ptr);
//     ASSERT_EQ(label_predicted.size(), params.value_ptr->size());
//     unsigned long long count = 0;
//     for (typename std::vector<real_type>::size_type i = 0; i < label_predicted.size(); ++i) {
//         if (label_predicted[i] == (*params.value_ptr)[i]) {
//             ++count;
//         }
//     }
//     const real_type accuracy_correct = static_cast<real_type>(count) / static_cast<real_type>(label_predicted.size());
//
//     // calculate accuracy using the intern data and labels
//     const real_type accuracy_calculated_intern = csvm.accuracy();
//     util::gtest_assert_floating_point_eq(accuracy_calculated_intern, accuracy_correct);
//     // calculate accuracy using external data and labels
//     const real_type accuracy_calculated_extern = csvm.accuracy(*params.data_ptr, *params.value_ptr);
//
//     util::gtest_assert_floating_point_eq(accuracy_calculated_extern, accuracy_correct);
//
//     // check single point prediction
//     const real_type prediction_first_point = csvm.predict_label((*params.data_ptr)[0]);
//     const real_type accuracy_calculated_single_point_correct = csvm.accuracy((*params.data_ptr)[0], prediction_first_point);
//     util::gtest_assert_floating_point_eq(accuracy_calculated_single_point_correct, real_type{ 1.0 });
//     const real_type accuracy_calculated_single_point_wrong = csvm.accuracy((*params.data_ptr)[0], -prediction_first_point);
//     util::gtest_assert_floating_point_eq(accuracy_calculated_single_point_wrong, real_type{ 0.0 });
// }

}  // namespace generic