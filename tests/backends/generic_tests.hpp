/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 *  @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic tests for all backends to reduce code duplication.
 */

#pragma once

#include "../mock_csvm.hpp"  // mock_csvm
#include "../utility.hpp"    // util::gtest_assert_floating_point_near, util::gtest_assert_floating_point_eq, util::gtest_expect_correct_csvm_factory, util::create_temp_file
#include "compare.hpp"       // compare::generate_q, compare::kernel_function, compare::device_kernel_function

#include "plssvm/backend_types.hpp"             // plssvm::backend_type
#include "plssvm/constants.hpp"                 // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/kernel_types.hpp"              // plssvm::kernel_type
#include "plssvm/parameter.hpp"                 // plssvm::parameter

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // GTEST_USES_POSIX_RE, EXPECT_THAT, ASSERT_EQ, EXPECT_EQ, EXPECT_GT, testing::ContainsRegex, testing::StaticAssertTypeEq

#include <algorithm>   // std::generate
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>      // std::string, std::getline
#include <vector>      // std::vector

namespace generic {

template <template <typename> typename csvm_type, typename real_type, plssvm::backend_type backend>
inline void csvm_factory_test() {
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.backend = backend;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    util::gtest_expect_correct_csvm_factory<csvm_type>(params);
}

template <template <typename> typename csvm_type, typename real_type, plssvm::kernel_type kernel>
inline void write_model_test() {
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.kernel = kernel;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM based on specified backend
    csvm_type csvm{ params };

    // create temporary model file and write model
    std::string model_file = util::create_temp_file();

    // learn model
    csvm.learn();

    // write learned model to file
    csvm.write_model(model_file);

    // read content of model file and delete it
    std::ifstream model_ifs(model_file);
    std::string file_content((std::istreambuf_iterator<char>(model_ifs)), std::istreambuf_iterator<char>());
    model_ifs.close();
    std::filesystem::remove(model_file);

    // check model file content for correctness
#ifdef GTEST_USES_POSIX_RE
    switch (params.kernel) {
        case plssvm::kernel_type::linear:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
        case plssvm::kernel_type::polynomial:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type polynomial\ndegree [0-9]+\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\ncoef0 [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
        case plssvm::kernel_type::rbf:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type rbf\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
    }
#endif
}

template <template <typename> typename csvm_type, typename real_type, plssvm::kernel_type kernel>
inline void generate_q_test() {
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.kernel = kernel;

    params.parse_train_file(TEST_FILE);

    // create base C-SVM
    mock_csvm csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    // parse libsvm file and calculate q vector
    const std::vector<real_type> correct = compare::generate_q<kernel>(csvm.get_data(), csvm);

    // setup C-SVM based on specified backend
    csvm_type csvm_backend{ params };
    using real_type_csvm_backend = typename decltype(csvm_backend)::real_type;

    // check types
    testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_backend>();

    // calculate q vector
    csvm_backend.setup_data_on_device();
    const std::vector<real_type> calculated = csvm_backend.generate_q();

    ASSERT_EQ(correct.size(), calculated.size());
    for (typename std::vector<real_type>::size_type index = 0; index < correct.size(); ++index) {
        util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}", index));
    }
}

template <template <typename> typename csvm_type, typename real_type, plssvm::kernel_type kernel>
inline void device_kernel_test() {
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.kernel = kernel;

    params.parse_train_file(TEST_FILE);

    // create base C-SVM
    mock_csvm csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    const std::size_t dept = csvm.get_num_data_points() - 1;

    // create x vector and fill it with random values
    std::vector<real_type> x(dept);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(1.0, 2.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

    // create correct q vector, cost and QA_cost
    const std::vector<real_type> q_vec = compare::generate_q<kernel>(csvm.get_data(), csvm);
    const real_type cost = csvm.get_cost();
    const real_type QA_cost = compare::kernel_function<kernel>(csvm.get_data().back(), csvm.get_data().back(), csvm) + 1 / cost;

    // create C-SVM using the specified backend
    csvm_type csvm_backend{ params };
    using real_type_csvm_backend = typename decltype(csvm)::real_type;
    using device_ptr_type = typename decltype(csvm_backend)::device_ptr_type;
    using queue_type = typename decltype(csvm_backend)::queue_type;

    // check types
    testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_backend>();

    // setup data on device
    csvm_backend.setup_data_on_device();

    // setup all additional data
    const std::size_t boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    std::vector<device_ptr_type> q_d{};
    std::vector<device_ptr_type> x_d{};
    std::vector<device_ptr_type> r_d{};

    for (queue_type &queue : csvm_backend.get_devices()) {
        q_d.emplace_back(dept + boundary_size, queue).memcpy_to_device(q_vec, 0, dept);
        x_d.emplace_back(dept + boundary_size, queue).memcpy_to_device(x, 0, dept);
        r_d.emplace_back(dept + boundary_size, queue).memset(0);
    }

    for (const auto add : { real_type{ -1 }, real_type{ 1 } }) {
        const std::vector<real_type> correct = compare::device_kernel_function<kernel>(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm);

        csvm_backend.set_QA_cost(QA_cost);
        csvm_backend.set_cost(cost);

        for (typename std::vector<queue_type>::size_type device = 0; device < csvm_backend.get_num_devices(); ++device) {
            csvm_backend.run_device_kernel(device, q_d[device], r_d[device], x_d[device], add);
        }
        std::vector<real_type> calculated(dept);
        csvm_backend.device_reduction(r_d, calculated);

        for (device_ptr_type &r_d_device : r_d) {
            r_d_device.memset(0);
        }

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (typename std::vector<real_type>::size_type index = 0; index < correct.size(); ++index) {
            util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}, add: {}", index, add));
        }
    }
}

template <template <typename> typename csvm_type, typename real_type, plssvm::kernel_type kernel>
inline void predict_test() {
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;

    params.parse_model_file(fmt::format(TEST_PATH "/data/models/500x200.libsvm.{}.model", kernel));
    params.parse_test_file(TEST_PATH "/data/libsvm/500x200.libsvm.test");

    // create C-SVM using the specified backend
    csvm_type csvm{ params };

    // predict
    std::vector<real_type> predicted_values = csvm.predict_label(*params.test_data_ptr);
    std::vector<real_type> predicted_values_real = csvm.predict(*params.test_data_ptr);

    // read correct prediction
    std::ifstream ifs(TEST_PATH "/data/predict/500x200.libsvm.predict");
    std::string line;
    std::vector<real_type> correct_values;
    correct_values.reserve(500);
    while (std::getline(ifs, line, '\n')) {
        correct_values.push_back(plssvm::detail::convert_to<real_type>(line));
    }

    ASSERT_EQ(correct_values.size(), predicted_values.size());
    for (typename std::vector<real_type>::size_type i = 0; i < correct_values.size(); ++i) {
        EXPECT_EQ(correct_values[i], predicted_values[i]) << "data point: " << i << " real value: " << predicted_values_real[i];
        EXPECT_GT(correct_values[i] * predicted_values_real[i], real_type{ 0 });
    }

    // empty points should return an empty vector
    std::vector<std::vector<real_type>> points;
    EXPECT_EQ(std::vector<real_type>{}, csvm.predict(points));

    // test exceptions
    [[maybe_unused]] std::vector<real_type> acc_vec;
    // the number of features of the prediction points must all be the same
    points = { { real_type{ 1 }, real_type{ 2 } }, { real_type{ 3 } } };
    EXPECT_THROW_WHAT(acc_vec = csvm.predict(points), plssvm::exception, "All points in the prediction point vector must have the same number of features!");

    // the number of features of the prediction points must match the number of features of the data points
    points = { { real_type{ 1 }, real_type{ 2 } }, { real_type{ 3 }, real_type{ 4 } } };
    EXPECT_THROW_WHAT(acc_vec = csvm.predict_label(points), plssvm::exception, fmt::format("Number of features per data point ({}) must match the number of features per predict point (2)!", params.data_ptr->front().size()));

    // alpha values must not be nullptr in order to predict
    csvm.get_alpha_ptr() = nullptr;
    EXPECT_THROW_WHAT(acc_vec = csvm.predict_label(*params.test_data_ptr), plssvm::exception, "No alphas provided for prediction!");
}

template <template <typename> typename csvm_type, typename real_type, plssvm::kernel_type kernel>
inline void accuracy_test() {
    // create parameter object
    plssvm::parameter<real_type> params;
    params.print_info = false;
    params.kernel = kernel;

    params.parse_train_file(TEST_FILE);

    // create C-SVM using the specified backend
    csvm_type csvm{ params };

    // learn
    csvm.learn();

    // predict label and calculate correct accuracy
    const std::vector<real_type> label_predicted = csvm.predict_label(*params.data_ptr);
    ASSERT_EQ(label_predicted.size(), params.value_ptr->size());
    unsigned long long count = 0;
    for (typename std::vector<real_type>::size_type i = 0; i < label_predicted.size(); ++i) {
        if (label_predicted[i] == (*params.value_ptr)[i]) {
            ++count;
        }
    }
    const real_type accuracy_correct = static_cast<real_type>(count) / static_cast<real_type>(label_predicted.size());

    // calculate accuracy using the intern data and labels
    const real_type accuracy_calculated_intern = csvm.accuracy();
    util::gtest_assert_floating_point_eq(accuracy_calculated_intern, accuracy_correct);
    // calculate accuracy using external data and labels
    const real_type accuracy_calculated_extern = csvm.accuracy(*params.data_ptr, *params.value_ptr);

    util::gtest_assert_floating_point_eq(accuracy_calculated_extern, accuracy_correct);

    // check single point prediction
    const real_type prediction_first_point = csvm.predict_label((*params.data_ptr)[0]);
    const real_type accuracy_calculated_single_point_correct = csvm.accuracy((*params.data_ptr)[0], prediction_first_point);
    util::gtest_assert_floating_point_eq(accuracy_calculated_single_point_correct, real_type{ 1.0 });
    const real_type accuracy_calculated_single_point_wrong = csvm.accuracy((*params.data_ptr)[0], -prediction_first_point);
    util::gtest_assert_floating_point_eq(accuracy_calculated_single_point_wrong, real_type{ 0.0 });
}

}  // namespace generic