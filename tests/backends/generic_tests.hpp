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

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_TESTS_HPP_
#pragma once

#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/model.hpp"                  // plssvm::model
#include "plssvm/parameter.hpp"              // plssvm::detail::parameter

#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_VECTOR_NEAR

#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads
#include "utility.hpp"
#include "gtest/gtest.h"  // ASSERT_EQ, EXPECT_EQ, EXPECT_TRUE

#include <cmath>     // std::sqrt
#include <fstream>   // std::ifstream
#include <iterator>  // std::istream_iterator
#include <utility>   // std::pair
#include <vector>    // std::vector

namespace generic {

// TODO: add non-trivial test

template <typename real_type, typename mock_csvm_type>
inline void test_solve_system_of_linear_equations(const plssvm::kernel_function_type kernel) {
    // create parameter struct
    plssvm::detail::parameter<real_type> params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };
    if (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1.0;
        params.gamma = 1.0;
        params.coef0 = 0.0;
    }

    // create the data that should be used
    // Matrix with 1-1/cost on main diagonal. Thus, the diagonal entries become one with the additional addition of 1/cost
    const std::vector<std::vector<real_type>> A = {
        { real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ std::sqrt(real_type(1.0) - 1 / params.cost) } },
    };
    const std::vector<real_type> rhs{ real_type{ 1.0 }, real_type{ -1.0 }, real_type{ 1.0 }, real_type{ -1.0 } };

    // create C-SVM: must be done using the mock class, since solve_system_of_linear_equations_impl is protected
    const mock_csvm_type svm{};

    // solve the system of linear equations using the CG algorithm:
    // | Q  1 |  *  | a |  =  | y |
    // | 1  0 |     | b |     | 0 |
    // with Q = A^TA
    const auto &[calculated_x, calculated_rho] = svm.solve_system_of_linear_equations(params, A, rhs, real_type{ 0.00001 }, A.front().size());

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated_x, rhs);
    // EXPECT_FLOATING_POINT_NEAR(calculated_rho, real_type{ 0.0 });
    EXPECT_FLOATING_POINT_NEAR(std::abs(calculated_rho) - std::numeric_limits<real_type>::epsilon(), std::numeric_limits<real_type>::epsilon());
}

template <typename real_type, typename mock_csvm_type>
inline void test_predict_values(const plssvm::kernel_function_type kernel) {
    // create parameter struct
    plssvm::detail::parameter<real_type> params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };
    if (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1.0;
        params.gamma = 1.0;
        params.coef0 = 0.0;
    }

    // create the data that should be used
    const std::vector<std::vector<real_type>> support_vectors = {
        { real_type{ 1.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 0.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 0.0 } },
        { real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 0.0 }, real_type{ 1.0 } }
    };
    const std::vector<real_type> weights{ real_type{ 1.0 }, real_type{ -1.0 }, real_type{ 1.0 }, real_type{ -1.0 } };
    const real_type rho{ 0.0 };
    std::vector<real_type> w{};
    const std::vector<std::vector<real_type>> data{
        { real_type{ 1.0 }, real_type{ 1.0 }, real_type{ 1.0 }, real_type{ 1.0 } },
        { real_type{ 1.0 }, real_type{ -1.0 }, real_type{ 1.0 }, real_type{ -1.0 } }
    };

    // create C-SVM: must be done using the mock class, since solve_system_of_linear_equations_impl is protected
    const mock_csvm_type svm{};

    // predict the values using the previously learned support vectors and weights
    const std::vector<real_type> calculated = svm.predict_values(params, support_vectors, weights, rho, w, data);

    // check the calculated result for correctness
    ASSERT_EQ(calculated.size(), data.size());
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, (std::vector<real_type>{ real_type{ 0.0 }, real_type{ 4.0 } }));
    // in case of the linear kernel, the w vector should have been filled
    if (kernel == plssvm::kernel_function_type::linear) {
        EXPECT_EQ(w.size(), support_vectors.front().size());
        EXPECT_FLOATING_POINT_VECTOR_NEAR(w, weights);
    } else {
        EXPECT_TRUE(w.empty());
    }
}

template <typename real_type, typename mock_csvm_type>
inline void test_generate_q(const plssvm::kernel_function_type kernel_type) {
    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ kernel_type, 2, 0.001, 1.0, 0.1 };

    // create the data that should be used
    const plssvm::data_set<real_type> data{ PLSSVM_TEST_FILE };

    // calculate correct q vector (ground truth)
    const std::vector<real_type> ground_truth = compare::generate_q(params, data.data());

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::generate_q is protected
    const mock_csvm_type svm{};

    // perform the data setup on the device
    constexpr std::size_t boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    const std::size_t num_used_devices = svm.select_num_used_devices(params.kernel_type, data.num_features());
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(data.data(), data.num_data_points() - 1, data.num_features(), boundary_size, num_used_devices);

    // calculate the q vector using a GPU backend
    const std::vector<real_type> calculated = svm.generate_q(params, data_d, data_last_d, data.num_data_points() - 1, feature_ranges, boundary_size, num_used_devices);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(ground_truth, calculated);
}

template <typename real_type, typename mock_csvm_type>
inline void test_calculate_w() {
    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;
    using queue_type = typename mock_csvm_type::queue_type;

    // create the data that should be used
    const plssvm::data_set<real_type> support_vectors{ PLSSVM_TEST_FILE };
    const std::vector<real_type> weights = util::generate_random_vector<real_type>(support_vectors.num_data_points());

    // calculate the correct w vector
    const std::vector<real_type> ground_truth = compare::calculate_w(support_vectors.data(), weights);

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::calculate_w is protected
    const mock_csvm_type svm{};

    // perform the data setup on the device
    constexpr std::size_t boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    const std::size_t num_support_vectors = support_vectors.num_data_points();
    const std::size_t num_used_devices = svm.select_num_used_devices(plssvm::kernel_function_type::linear, support_vectors.num_features());
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(support_vectors.data(), num_support_vectors - 1, support_vectors.num_features(), boundary_size, num_used_devices);

    std::vector<device_ptr_type> alpha_d(num_used_devices);
    for (typename std::vector<queue_type>::size_type device = 0; device < num_used_devices; ++device) {
        alpha_d[device] = device_ptr_type{ num_support_vectors + plssvm::THREAD_BLOCK_SIZE, svm.devices_[device] };
        alpha_d[device].memset(0);
        alpha_d[device].memcpy_to_device(weights, 0, num_support_vectors);
    }

    // calculate the w vector using a GPU backend
    const std::vector<real_type> calculated = svm.calculate_w(data_d, data_last_d, alpha_d, num_support_vectors, feature_ranges, num_used_devices);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(ground_truth, calculated);
}

template <typename real_type, typename mock_csvm_type>
inline void test_run_device_kernel(const plssvm::kernel_function_type kernel_type) {
    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;
    using queue_type = typename mock_csvm_type::queue_type;

    // create parameter struct
    const plssvm::detail::parameter<real_type> params{ kernel_type, 2, 0.001, 1.0, 0.1 };

    // create the data that should be used
    const plssvm::data_set<real_type> data{ PLSSVM_TEST_FILE };
    const std::size_t dept = data.num_data_points() - 1;
    const std::vector<real_type> rhs = util::generate_random_vector<real_type>(dept, real_type{ 1.0 }, real_type{ 2.0 });
    const std::vector<real_type> q = compare::generate_q(params, data.data());
    const real_type QA_cost = compare::kernel_function(params, data.data().back(), data.data().back()) + 1 / params.cost;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::calculate_w is protected
    const mock_csvm_type svm{};

    // perform the data setup on the device
    constexpr std::size_t boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    const std::size_t num_used_devices = svm.select_num_used_devices(kernel_type, data.num_features());
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(data.data(), dept, data.num_features(), boundary_size, num_used_devices);
    std::vector<device_ptr_type> q_d{};
    std::vector<device_ptr_type> x_d{};
    std::vector<device_ptr_type> r_d{};

    for (const queue_type &queue : svm.devices_) {
        q_d.emplace_back(dept + boundary_size, queue).memcpy_to_device(q, 0, dept);
        x_d.emplace_back(dept + boundary_size, queue).memset(0);
        x_d.back().memcpy_to_device(rhs, 0, dept);
        r_d.emplace_back(dept + boundary_size, queue).memset(0);
    }

    for (const real_type add : { real_type{ -1.0 }, real_type{ 1.0 } }) {
        // calculate the correct device function result
        const std::vector<real_type> ground_truth = compare::device_kernel_function(params, data.data(), rhs, q, QA_cost, add);

        // perform the kernel calculation on the device
        for (std::size_t device = 0; device < num_used_devices; ++device) {
            svm.run_device_kernel(device, params, q_d[device], r_d[device], x_d[device], data_d[device], feature_ranges, QA_cost, add, dept, boundary_size);
        }
        std::vector<real_type> calculated(dept);
        svm.device_reduction(r_d, calculated);

        for (device_ptr_type &r_d_device : r_d) {
            r_d_device.memset(0);
        }

        // check the calculated result for correctness
        EXPECT_FLOATING_POINT_VECTOR_NEAR(ground_truth, calculated);
    }
}

template <typename real_type, typename mock_csvm_type>
inline void test_device_reduction() {
    using namespace plssvm::operators;

    using device_ptr_type = typename mock_csvm_type::template device_ptr_type<real_type>;
    using queue_type = typename mock_csvm_type::queue_type;

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::device_reduction is protected
    const mock_csvm_type svm{};

    // data to reduce
    const std::vector<real_type> data(1'024, real_type{ 1.0 });

    // perform the kernel calculation on the device
    std::vector<device_ptr_type> data_d{};
    for (const queue_type &queue : svm.devices_) {
        data_d.emplace_back(data.size(), queue).memcpy_to_device(data);
    }

    std::vector<real_type> calculated(data.size());
    svm.device_reduction(data_d, calculated);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, data * static_cast<real_type>(svm.devices_.size()));

    // check if the values have been correctly promoted back onto the device(s)
    for (std::size_t device = 0; device < svm.devices_.size(); ++device) {
        std::vector<real_type> device_data(data.size());
        data_d[device].memcpy_to_host(device_data);
        EXPECT_FLOATING_POINT_VECTOR_NEAR(device_data, data * static_cast<real_type>(svm.devices_.size()));
    }

    //
    // test reduction of a single device
    data_d.resize(1);
    calculated.assign(data.size(), real_type{ 0.0 });
    svm.device_reduction(data_d, calculated);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_VECTOR_NEAR(calculated, data);
    std::vector<real_type> device_data(data.size());
    data_d.front().memcpy_to_host(device_data);
    EXPECT_FLOATING_POINT_VECTOR_NEAR(device_data, data);
}

template <typename real_type, typename mock_csvm_type>
inline void test_select_num_used_devices(const plssvm::kernel_function_type kernel_type) {
    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::select_num_used_devices is protected
    const mock_csvm_type svm{};

    if (kernel_type == plssvm::kernel_function_type::linear) {
        // the linear kernel function is compatible with at least one device
        EXPECT_GE(svm.select_num_used_devices(kernel_type, 1'204), 1);
    } else {
        // the other kernel functions are compatible only with exactly one device
        EXPECT_EQ(svm.select_num_used_devices(kernel_type, 1'204), 1);
    }
    // if only a single feature is provided, only a single device may ever be used
    EXPECT_EQ(svm.select_num_used_devices(kernel_type, 1), 1);
}

template <typename real_type, typename mock_csvm_type>
inline void test_setup_data_on_device_minimal() {
    // minimal example
    const std::vector<std::vector<real_type>> input = {
        { real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 }, real_type{ 6.0 } },
        { real_type{ 7.0 }, real_type{ 8.0 }, real_type{ 9.0 } },
    };

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::setup_data_on_device is protected
    const mock_csvm_type svm{};

    // perform data setup on the device
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(input, input.size(), input.front().size(), 0, 1);

    // check returned values
    // the "big" data vector
    ASSERT_EQ(data_d.size(), 1);
    ASSERT_EQ(data_d[0].size(), 9);
    std::vector<real_type> data(9);
    data_d[0].memcpy_to_host(data);
    EXPECT_FLOATING_POINT_VECTOR_EQ(data, (std::vector<real_type>{ real_type{ 1.0 }, real_type{ 4.0 }, real_type{ 7.0 }, real_type{ 2.0 }, real_type{ 5.0 }, real_type{ 8.0 }, real_type{ 3.0 }, real_type{ 6.0 }, real_type{ 9.0 } }));
    // the last row in the original data
    ASSERT_EQ(data_last_d.size(), 1);
    ASSERT_EQ(data_last_d[0].size(), 3);
    std::vector<real_type> data_last(3);
    data_last_d[0].memcpy_to_host(data_last);
    EXPECT_FLOATING_POINT_VECTOR_EQ(data_last, (std::vector<real_type>{ real_type{ 7.0 }, real_type{ 8.0 }, real_type{ 9.0 } }));
    // the feature ranges
    ASSERT_EQ(feature_ranges.size(), 2);
    EXPECT_EQ(feature_ranges, (std::vector<std::size_t>{ 0, input.front().size() }));
}

template <typename real_type, typename mock_csvm_type>
inline void test_setup_data_on_device() {
    // minimal example
    const std::vector<std::vector<real_type>> input = {
        { real_type{ 1.0 }, real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 }, real_type{ 6.0 } },
        { real_type{ 7.0 }, real_type{ 8.0 }, real_type{ 9.0 } },
    };

    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::setup_data_on_device is protected
    const mock_csvm_type svm{};

    // perform data setup on the device
    const std::size_t boundary_size = 2;
    const std::size_t num_devices = svm.select_num_used_devices(plssvm::kernel_function_type::linear, input.front().size());
    const std::size_t num_data_points = input.size();
    const std::size_t num_features = input.front().size();
    auto [data_d, data_last_d, feature_ranges] = svm.setup_data_on_device(input, num_data_points - 1, num_features, boundary_size, num_devices);

    // check returned values
    // the feature ranges
    ASSERT_EQ(feature_ranges.size(), num_devices + 1);
    for (std::size_t i = 0; i <= num_devices; ++i) {
        EXPECT_EQ(feature_ranges[i], i * num_features / num_devices);
    }

    const std::vector<real_type> transformed_data = plssvm::detail::transform_to_layout(plssvm::detail::layout_type::soa, input, boundary_size, num_data_points - 1);

    // the "big" data vector
    ASSERT_EQ(data_d.size(), num_devices);
    for (std::size_t device = 0; device < num_devices; ++device) {
        const std::size_t expected_size = (feature_ranges[device + 1] - feature_ranges[device]) * (num_data_points - 1 + boundary_size);
        ASSERT_EQ(data_d[device].size(), expected_size) << fmt::format("for device {}: [{}, {}]", device, feature_ranges[device], feature_ranges[device + 1]);
        std::vector<real_type> data(expected_size);
        data_d[device].memcpy_to_host(data);

        const std::size_t device_data_size = (feature_ranges[device + 1] - feature_ranges[device]) * (num_data_points - 1 + boundary_size);
        const std::vector<real_type> ground_truth(transformed_data.data() + feature_ranges[device] * (num_data_points - 1 + boundary_size),
                                                  transformed_data.data() + feature_ranges[device] * (num_data_points - 1 + boundary_size) + device_data_size);
        EXPECT_FLOATING_POINT_VECTOR_EQ(data, ground_truth);
    }
    // the last row in the original data
    ASSERT_EQ(data_last_d.size(), num_devices);
    for (std::size_t device = 0; device < num_devices; ++device) {
        const std::size_t expected_size = feature_ranges[device + 1] - feature_ranges[device] + boundary_size;
        ASSERT_EQ(data_last_d[device].size(), expected_size);
        std::vector<real_type> data_last(expected_size);
        data_last_d[device].memcpy_to_host(data_last);

        std::vector<real_type> ground_truth(input.back().data() + feature_ranges[device], input.back().data() + feature_ranges[device] + (feature_ranges[device + 1] - feature_ranges[device]));
        ground_truth.resize(ground_truth.size() + boundary_size, real_type{ 0.0 });
        EXPECT_FLOATING_POINT_VECTOR_EQ(data_last, ground_truth);
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

template <typename mock_csvm_type>
inline void test_num_available_devices() {
    // create C-SVM: must be done using the mock class, since plssvm::detail::gpu_csvm::devices_ is protected
    const mock_csvm_type svm{};

    // test the number of available devices
    ASSERT_EQ(svm.num_available_devices(), svm.devices_.size());
    EXPECT_GE(svm.num_available_devices(), 1);
}

}  // namespace generic

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_TESTS_HPP_