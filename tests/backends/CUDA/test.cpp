/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the CUDA backend.
 */

#include "mock_cuda_csvm.hpp"

#include "../../mock_csvm.hpp"  // mock_csvm
#include "../../utility.hpp"    // util::create_temp_file, util::gtest_expect_correct_csvm_factory, EXPECT_THROW_WHAT,
                                // util::google_test::parameter_definition, util::google_test::parameter_definition_to_name,
                                // util::gtest_assert_floating_point_near, util::gtest_assert_floating_point_eq
#include "../compare.hpp"       // compare::generate_q, compare::kernel_function, compare::device_kernel_function

#include "plssvm/backend_types.hpp"             // plssvm::backend_type
#include "plssvm/backends/CUDA/csvm.hpp"        // plssvm::cuda::csvm
#include "plssvm/backends/CUDA/exceptions.hpp"  // plssvm::cuda::backend_exception
#include "plssvm/constants.hpp"                 // plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/kernel_types.hpp"              // plssvm::kernel_type
#include "plssvm/parameter.hpp"                 // plssvm::parameter

#include "gtest/gtest.h"  // ::testing::StaticAssertTypeEq, ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST,
                          // ASSERT_EQ, EXPECT_EQ, EXPECT_GT, EXPECT_NO_THROW, EXPECT_THAT

#include <algorithm>   // std::generate
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>      // std::string, std::getline
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
class CUDA_CSVM : public ::testing::Test {};
TYPED_TEST_SUITE(CUDA_CSVM, parameter_types, util::google_test::parameter_definition_to_name);

// check whether the csvm factory function correctly creates a cuda::csvm
TYPED_TEST(CUDA_CSVM, csvm_factory) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.backend = plssvm::backend_type::cuda;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    util::gtest_expect_correct_csvm_factory<plssvm::cuda::csvm>(params);
}

// check whether the constructor correctly fails when using an incompatible target platform
TYPED_TEST(CUDA_CSVM, constructor_invalid_target_platform) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    // only automatic or gpu_nvidia are allowed as target platform for the CUDA backend
    params.target = plssvm::target_platform::automatic;
    EXPECT_NO_THROW(mock_cuda_csvm{ params });
    params.target = plssvm::target_platform::cpu;
    EXPECT_THROW_WHAT(mock_cuda_csvm{ params }, plssvm::cuda::backend_exception, "Invalid target platform 'cpu' for the CUDA backend!");
    params.target = plssvm::target_platform::gpu_nvidia;
    EXPECT_NO_THROW(mock_cuda_csvm{ params });
    params.target = plssvm::target_platform::gpu_amd;
    EXPECT_THROW_WHAT(mock_cuda_csvm{ params }, plssvm::cuda::backend_exception, "Invalid target platform 'gpu_amd' for the CUDA backend!");
    params.target = plssvm::target_platform::gpu_intel;
    EXPECT_THROW_WHAT(mock_cuda_csvm{ params }, plssvm::cuda::backend_exception, "Invalid target platform 'gpu_intel' for the CUDA backend!");
}

// check whether writing the resulting model file is correct
TYPED_TEST(CUDA_CSVM, write_model) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_cuda_csvm csvm{ params };

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

// check whether the q vector is generated correctly
TYPED_TEST(CUDA_CSVM, generate_q) {
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

    // create C-SVM using the CUDA backend
    mock_cuda_csvm csvm_cuda{ params };
    using real_type_csvm_cuda = typename decltype(csvm_cuda)::real_type;

    // check real_types
    ::testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_cuda>();

    // calculate q vector
    csvm_cuda.setup_data_on_device();
    const std::vector<real_type_csvm_cuda> calculated = csvm_cuda.generate_q();

    ASSERT_EQ(correct.size(), calculated.size());
    for (std::size_t index = 0; index < correct.size(); ++index) {
        util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}", index));
    }
}

// check whether the device kernels are correct
TYPED_TEST(CUDA_CSVM, device_kernel) {
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

    // create C-SVM using the CUDA backend
    mock_cuda_csvm csvm_cuda{ params };

    // setup data on device
    csvm_cuda.setup_data_on_device();

    // setup all additional data
    const size_type boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    std::vector<plssvm::cuda::detail::device_ptr<real_type>> q_d(csvm_cuda.get_num_devices());
    std::vector<plssvm::cuda::detail::device_ptr<real_type>> x_d(csvm_cuda.get_num_devices());
    std::vector<plssvm::cuda::detail::device_ptr<real_type>> r_d(csvm_cuda.get_num_devices());

    for (int device = 0; device < csvm_cuda.get_num_devices(); ++device) {
        q_d[device] = plssvm::cuda::detail::device_ptr<real_type>{ dept + boundary_size, device };
        q_d[device].memcpy_to_device(q_vec, 0, dept);
        x_d[device] = plssvm::cuda::detail::device_ptr<real_type>{ dept + boundary_size, device };
        x_d[device].memcpy_to_device(x, 0, dept);
        r_d[device] = plssvm::cuda::detail::device_ptr<real_type>{ dept + boundary_size, device };
        r_d[device].memset(0);
    }

    for (const auto add : { real_type{ -1 }, real_type{ 1 } }) {
        const std::vector<real_type> correct = compare::device_kernel_function<TypeParam::kernel>(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm);

        csvm_cuda.set_QA_cost(QA_cost);
        csvm_cuda.set_cost(cost);

        for (int device = 0; device < csvm_cuda.get_num_devices(); ++device) {
            csvm_cuda.run_device_kernel(device, q_d[device], r_d[device], x_d[device], add);
        }
        std::vector<real_type> calculated(dept);
        csvm_cuda.device_reduction(r_d, calculated);

        for (plssvm::cuda::detail::device_ptr<real_type> &r_d_device : r_d) {
            r_d_device.memset(0);
        }

        ASSERT_EQ(correct.size(), calculated.size()) << "add: " << add;
        for (std::size_t index = 0; index < correct.size(); ++index) {
            util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}, add: {}", index, add));
        }
    }
}

// check whether the correct labels are predicted
TYPED_TEST(CUDA_CSVM, predict) {
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;

    params.parse_model_file(TEST_PATH "/data/models/500x200.libsvm." + fmt::format("{}", TypeParam::kernel) + ".model");
    params.parse_test_file(TEST_PATH "/data/libsvm/500x200.libsvm.test");

    // create C-SVM using the CUDA backend
    mock_cuda_csvm csvm_cuda{ params };
    using real_type = typename decltype(csvm_cuda)::real_type;
    using size_type = typename decltype(csvm_cuda)::size_type;

    // predict
    std::vector<real_type> predicted_values = csvm_cuda.predict_label(*params.test_data_ptr);
    std::vector<real_type> predicted_values_real = csvm_cuda.predict(*params.test_data_ptr);

    // read correct prediction
    std::ifstream ifs(fmt::format("{}{}", TEST_PATH, "/data/predict/500x200.libsvm.predict"));
    std::string line;
    std::vector<real_type> correct_values;
    correct_values.reserve(500);
    while (std::getline(ifs, line, '\n')) {
        correct_values.push_back(plssvm::detail::convert_to<real_type>(line));
    }

    ASSERT_EQ(correct_values.size(), predicted_values.size());
    for (size_type i = 0; i < correct_values.size(); ++i) {
        EXPECT_EQ(correct_values[i], predicted_values[i]) << "data point: " << i << " real value: " << predicted_values_real[i];
        EXPECT_GT(correct_values[i] * predicted_values_real[i], real_type{ 0 });
    }
}

// check whether the accuracy calculation is correct
TYPED_TEST(CUDA_CSVM, accuracy) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_FILE);

    // create C-SVM using the CUDA backend
    mock_cuda_csvm csvm_cuda{ params };
    using real_type = typename decltype(csvm_cuda)::real_type;
    using size_type = typename decltype(csvm_cuda)::size_type;

    // learn
    csvm_cuda.learn();

    // predict label and calculate correct accuracy
    std::vector<real_type> label_predicted = csvm_cuda.predict_label(*params.data_ptr);
    ASSERT_EQ(label_predicted.size(), params.value_ptr->size());
    size_type count = 0;
    for (size_type i = 0; i < label_predicted.size(); ++i) {
        if (label_predicted[i] == (*params.value_ptr)[i]) {
            ++count;
        }
    }
    const real_type accuracy_correct = static_cast<real_type>(count) / static_cast<real_type>(label_predicted.size());

    // calculate accuracy using the intern data and labels
    const real_type accuracy_calculated_intern = csvm_cuda.accuracy();
    util::gtest_assert_floating_point_eq(accuracy_calculated_intern, accuracy_correct);
    // calculate accuracy using external data and labels
    const real_type accuracy_calculated_extern = csvm_cuda.accuracy(*params.data_ptr, *params.value_ptr);

    util::gtest_assert_floating_point_eq(accuracy_calculated_extern, accuracy_correct);

    // check single point prediction
    const real_type prediction_first_point = csvm_cuda.predict_label((*params.data_ptr)[0]);
    const real_type accuracy_calculated_single_point_correct = csvm_cuda.accuracy((*params.data_ptr)[0], prediction_first_point);
    util::gtest_assert_floating_point_eq(accuracy_calculated_single_point_correct, real_type{ 1.0 });
    const real_type accuracy_calculated_single_point_wrong = csvm_cuda.accuracy((*params.data_ptr)[0], -prediction_first_point);
    util::gtest_assert_floating_point_eq(accuracy_calculated_single_point_wrong, real_type{ 0.0 });
}
