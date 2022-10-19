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

#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/model.hpp"                  // plssvm::model
#include "plssvm/parameter.hpp"              // plssvm::detail::parameter

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_VECTOR_NEAR

#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads
#include "gtest/gtest.h"  // ASSERT_EQ, EXPECT_EQ, EXPECT_TRUE

#include <cmath>     // std::sqrt
#include <fstream>   // std::ifstream
#include <iterator>  // std::istream_iterator
#include <utility>   // std::pair
#include <vector>    // std::vector

namespace generic {

template <typename real_type, typename mock_csvm_type>
inline void test_solve_system_of_linear_equations(const plssvm::kernel_function_type kernel) {
    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ plssvm::kernel_type = kernel, plssvm::gamma = 0.2, plssvm::cost = 2.0 };

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

    // solve the system of linear equations using the CG algorithm: (A^TA + (I * 1 / cost))x = b  => Ix = b => should be trivial x = b
    const std::pair<std::vector<real_type>, real_type> calculated = svm.solve_system_of_linear_equations(params, data, rhs, real_type{ 0.00001 }, 1);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(rhs, calculated.first);
    // TODO: test rho
}

template <typename real_type, typename mock_csvm_type>
inline void test_predict_values(const plssvm::kernel_function_type kernel) {
    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ plssvm::kernel_type = kernel };

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
    const std::vector<real_type> calculated = svm.predict_values(params, support_vectors, weights, rho, w, data);

    // check the calculated result for correctness
    ASSERT_EQ(calculated.size(), data.size());
    // TODO: add other tests
    // in case of the linear kernel, the w vector should have been filled
    if (kernel == plssvm::kernel_function_type::linear) {
        EXPECT_EQ(w.size(), support_vectors.front().size());
    } else {
        EXPECT_TRUE(w.empty());
    }
}

template <typename real_type, typename csvm_type>
inline void test_predict(const plssvm::kernel_function_type kernel) {
    // create parameter struct
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create data set to be used
    const plssvm::data_set<real_type> test_data{ PLSSVM_TEST_PATH "/data/predict/500x200_test.libsvm" };

    // read the previously learned model
    const plssvm::model<real_type> model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/500x200_{}.libsvm.model", kernel) };

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
inline void test_score(const plssvm::kernel_function_type kernel) {
    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create data set to be used
    const plssvm::data_set<real_type> test_data{ PLSSVM_TEST_PATH "/data/predict/500x200_test.libsvm" };

    // read the previously learned model
    const plssvm::model<real_type> model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/500x200_{}.libsvm.model", kernel) };

    // create C-SVM
    const csvm_type svm{ params };

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

}  // namespace generic