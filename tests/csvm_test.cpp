/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Tests for the base functionality independent of the used backend.
 */

#include "mock_csvm.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception, plssvm::file_not_found_exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "backends/compare.hpp"  // compare::kernel_function
#include "utility.hpp"           // util::create_temp_file, util::gtest_expect_floating_point_eq

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ, EXPECT_EQ, EXPECT_THAT, EXPECT_THROW_WHAT, EXPECT_CALL

#include <algorithm>   // std::generate
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <iterator>    // std::istreambuf_iterator
#include <memory>      // std::make_shared
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>      // std::string
#include <vector>      // std::vector

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;

template <typename T>
class BaseCSVMTransform : public ::testing::Test {};
TYPED_TEST_SUITE(BaseCSVMTransform, floating_point_types);

// check whether transforming the 2D data into a 1D vector works as intended
TYPED_TEST(BaseCSVMTransform, transform_data) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;
    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    // transform data without and with boundary
    std::vector<real_type> result_no_boundary = csvm.transform_data(csvm.get_data(), 0, csvm.get_num_data_points() - 1);
    std::vector<real_type> result_boundary = csvm.transform_data(csvm.get_data(), 10, csvm.get_num_data_points());

    // check if sizes match
    ASSERT_EQ(result_no_boundary.size(), (csvm.get_num_data_points() - 1) * csvm.get_num_features());
    ASSERT_EQ(result_boundary.size(), (csvm.get_num_data_points() - 1 + 10) * csvm.get_num_features());

    // check transformed content for correctness
    for (size_type datapoint = 0; datapoint < csvm.get_num_data_points() - 1; ++datapoint) {
        for (size_type feature = 0; feature < csvm.get_num_features(); ++feature) {
            util::gtest_expect_floating_point_eq(
                csvm.get_data()[datapoint][feature],
                result_no_boundary[datapoint + feature * (csvm.get_num_data_points() - 1)],
                fmt::format("datapoint: {} feature: {} at index: {}", datapoint, feature, datapoint + feature * (csvm.get_num_data_points() - 1)));

            util::gtest_expect_floating_point_eq(
                csvm.get_data()[datapoint][feature],
                result_boundary[datapoint + feature * (csvm.get_num_data_points() - 1 + 10)],
                fmt::format("datapoint: {} feature: {} at index: {}", datapoint, feature, datapoint + feature * (csvm.get_num_data_points() + 10)));
        }
    }
}

// enumerate all floating point type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

template <typename T>
class BaseCSVM : public ::testing::Test {};
TYPED_TEST_SUITE(BaseCSVM, parameter_types, util::google_test::parameter_definition_to_name);

// check whether the plssvm::csvm<T>::csvm(plssvm::parameter<T>) constructor works as intended
TYPED_TEST(BaseCSVM, constructor) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;

    // check if constructor correctly initializes all fields
    EXPECT_EQ(params.target, csvm.get_target());
    EXPECT_EQ(params.kernel, csvm.get_kernel());
    EXPECT_EQ(params.degree, csvm.get_degree());
    EXPECT_EQ(params.gamma, csvm.get_gamma());
    EXPECT_EQ(params.coef0, csvm.get_coef0());
    EXPECT_EQ(params.cost, csvm.get_cost());
    EXPECT_EQ(params.epsilon, csvm.get_epsilon());
    EXPECT_EQ(params.print_info, csvm.get_print_info());

    EXPECT_EQ(params.data_ptr, csvm.get_data_ptr());
    EXPECT_EQ(params.value_ptr, csvm.get_value_ptr());
    EXPECT_EQ(params.alphas_ptr, csvm.get_alpha_ptr());

    EXPECT_EQ(params.data_ptr->size(), csvm.get_num_data_points());
    EXPECT_EQ(params.data_ptr->front().size(), csvm.get_num_features());
    EXPECT_EQ(-params.rho, csvm.get_bias());
    EXPECT_EQ(real_type{ 0 }, csvm.get_QA_cost());
}

// check whether calling the plssvm::csvm<T>::csvm(plssvm::parameter<T>) constructor without data correctly fails
TYPED_TEST(BaseCSVM, constructor_missing_data) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;

    // create C-SVM
    EXPECT_THROW_WHAT(mock_csvm csvm{ params }, plssvm::exception, "No data points provided!");
}

// check whether writing the resulting model file is correct
TYPED_TEST(BaseCSVM, write_model) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    // create temporary model file and write model
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
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type linear\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
        case plssvm::kernel_type::polynomial:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type polynomial\ndegree [0-9]+\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\ncoef0 [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
        case plssvm::kernel_type::rbf:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type rbf\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
    }
}

// check whether attempting to write the model file with missing data correctly fails
TYPED_TEST(BaseCSVM, write_model_missing_data) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    // attempting to write a model before a call to learn() should result in an exception
    EXPECT_THROW_WHAT(csvm.write_model("foo"), plssvm::exception, "No alphas given! Maybe a call to 'learn()' is missing?");

    // attempting to write a model with da data set without labels should result in an exception
    csvm.get_alpha_ptr() = std::make_shared<const std::vector<typename TypeParam::real_type>>();
    csvm.get_value_ptr() = nullptr;
    EXPECT_THROW_WHAT(csvm.write_model("foo"), plssvm::exception, "No labels given! Maybe the data is only usable for prediction?");
}

// check whether the correct kernel function is executed
TYPED_TEST(BaseCSVM, kernel_function) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    // set dummy data
    params.data_ptr = std::make_shared<const std::vector<std::vector<typename decltype(params)::real_type>>>(1);

    // create C-SVM
    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;

    // create dummy data vectors
    constexpr std::size_t size = 512;
    std::vector<real_type> x1(size);
    std::vector<real_type> x2(size);

    // fill vectors with random values
    std::random_device rnd_device;
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(1.0, 2.0);
    std::generate(x1.begin(), x1.end(), [&]() { return dist(gen); });
    std::generate(x2.begin(), x2.end(), [&]() { return dist(gen); });

    // calculated result
    const real_type calculated = csvm.kernel_function(x1, x2);
    // correct result
    const real_type correct = compare::kernel_function<TypeParam::kernel>(x1, x2, csvm);

    // check for correctness
    util::gtest_expect_floating_point_eq(correct, calculated);
}

// check whether plssvm::csvm<T>::learn() internally calls the correct functions
TYPED_TEST(BaseCSVM, learn) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    params.parse_train_file(TEST_PATH "/data/libsvm/5x4.libsvm");

    // create C-SVM
    mock_csvm csvm{ params };

    EXPECT_CALL(csvm, setup_data_on_device).Times(1);
    EXPECT_CALL(csvm, generate_q).Times(1);
    EXPECT_CALL(csvm, solver_CG).Times(1);

    csvm.learn();
}