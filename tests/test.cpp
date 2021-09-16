/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Tests for the base functionality independent of the used backend.
 */

#include "mock_csvm.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::replace_all
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception, plssvm::file_not_found_exception
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type
#include "plssvm/parameter_predict.hpp"      // plssvm::parameter_predict
#include "plssvm/parameter_train.hpp"        // plssvm::parameter_train

#include "backends/compare.hpp"  // linear_kernel
#include "utility.hpp"           // util::create_temp_file, util::gtest_expect_floating_point_eq

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ, EXPECT_EQ, EXPECT_THAT, EXPECT_THROW

#include <algorithm>   // std::generate
#include <cstddef>     // std::size_t
#include <filesystem>  // std::filesystem::remove
#include <fstream>     // std::ifstream
#include <iterator>    // std::istreambuf_iterator
#include <memory>      // std::make_shared
#include <random>      // std::random_device, std::mt19937, std::uniform_real_distribution
#include <sstream>     // std::stringstream
#include <string>      // std::string
#include <vector>      // std::vector

template <typename T>
class BASE : public ::testing::Test {};

using testing_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(BASE, testing_types);

TYPED_TEST(BASE, parse_libsvm) {
    // setup C-SVM class
    plssvm::parameter<TypeParam> params;
    params.print_info = false;
    params.parse_libsvm(TEST_PATH "/data/5x4.libsvm", params.data_ptr);

    using real_type = typename decltype(params)::real_type;
    using size_type = typename decltype(params)::size_type;

    // check if sizes match
    ASSERT_EQ(params.data_ptr->size(), 5) << "num data points mismatch";
    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(params.data_ptr->operator[](i).size(), 4) << "mismatch num features in datapoint: " << i;
    }

    // correct values
    std::vector<std::vector<real_type>> expected{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { -0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827 },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };

    // check parsed values for correctness
    for (size_type i = 0; i < 5; ++i) {
        for (size_type j = 0; j < 4; ++j) {
            util::gtest_expect_floating_point_eq((*(params.data_ptr))[i][j], expected[i][j], fmt::format("datapoint: {} feature: {}", i, j));
        }
    }
}

TYPED_TEST(BASE, parse_libsvm_sparse) {
    // setup C-SVM class
    plssvm::parameter<TypeParam> params;
    params.print_info = false;
    params.parse_libsvm(TEST_PATH "/data/5x4.sparse.libsvm", params.data_ptr);

    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    // check if sizes match
    ASSERT_EQ(params.data_ptr->size(), 5) << "num datapoints mismatch";
    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(params.data_ptr->operator[](i).size(), 4) << "mismatch num features in datapoint: " << i;
    }

    // correct values
    std::vector<std::vector<real_type>> expected{
        { 0., 0., 0., 0. },
        { 0., 0., 0.51687296029754564, 0. },
        { 0., 1.01405596624706053, 0., 0. },
        { 0., 0.60276937379453293, 0., -0.13086851759108944 },
        { 0., 0., 0.298499933047586044, 0. },
    };

    // check parsed values for correctness
    for (size_type i = 0; i < 5; ++i) {
        for (size_type j = 0; j < 4; ++j) {
            util::gtest_expect_floating_point_eq((*(params.data_ptr))[i][j], expected[i][j], fmt::format("datapoint: {} feature: {}", i, j));
        }
    }
}

TYPED_TEST(BASE, parse_arff) {
    // setup C-SVM class
    plssvm::parameter<TypeParam> params;
    params.print_info = false;
    params.parse_arff(TEST_PATH "/data/5x4.arff", params.data_ptr);

    using real_type = typename decltype(params)::real_type;
    using size_type = typename decltype(params)::size_type;

    // check if sizes match
    ASSERT_EQ(params.data_ptr->size(), 5) << "num datapoints mismatch";
    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(params.data_ptr->operator[](i).size(), 4) << "mismatch num features in datapoint: " << i;
    }

    // correct values
    std::vector<std::vector<real_type>> expected{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { 0., 0.60276937379453293, -0.13086851759108944, 0. },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };

    // check parsed values for correctness
    for (size_type i = 0; i < 5; ++i) {
        for (size_type j = 0; j < 4; ++j) {
            util::gtest_expect_floating_point_eq((*(params.data_ptr))[i][j], expected[i][j], fmt::format("datapoint: {} feature: {}", i, j));
        }
    }
}

TYPED_TEST(BASE, parse_libsvm_gamma) {
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // gamma = 1.0 (!= 0.0)
    params.gamma = 1.0;
    params.parse_libsvm(TEST_PATH "/data/5x4.libsvm", params.data_ptr);

    using real_type = typename decltype(params)::real_type;

    util::gtest_assert_floating_point_eq(real_type{ 1.0 }, params.gamma);

    // gamma = 0.0 -> automatically set to (1.0 / num_features)
    params.gamma = 0.0;
    params.parse_libsvm(TEST_PATH "/data/5x4.libsvm", params.data_ptr);

    util::gtest_assert_floating_point_eq(real_type{ 1.0 } / 4, params.gamma);
}

TYPED_TEST(BASE, parse_arff_gamma) {
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // gamma = 1.0 (!= 0.0)
    params.gamma = 1.0;
    params.parse_arff(TEST_PATH "/data/5x4.arff", params.data_ptr);

    using real_type = typename decltype(params)::real_type;

    util::gtest_assert_floating_point_eq(real_type{ 1.0 }, params.gamma);

    // gamma = 0.0 -> automatically set to (1.0 / num_features)
    params.gamma = 0.0;
    params.parse_arff(TEST_PATH "/data/5x4.arff", params.data_ptr);

    util::gtest_assert_floating_point_eq(real_type{ 1.0 } / 4, params.gamma);
}

TYPED_TEST(BASE, parse_file) {
    // setup C-SVM class
    plssvm::parameter<TypeParam> params;
    params.print_info = false;
    params.parse_train_file(TEST_PATH "/data/5x4.libsvm");

    using real_type = typename decltype(params)::real_type;
    using size_type = typename decltype(params)::size_type;

    // check if sizes match
    ASSERT_EQ(params.data_ptr->size(), 5) << "num data points mismatch";
    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(params.data_ptr->operator[](i).size(), 4) << "mismatch num features in datapoint: " << i;
    }

    // correct values
    std::vector<std::vector<real_type>> expected_libsvm{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { -0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827 },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };

    // check parsed values for correctness
    for (size_type i = 0; i < 5; ++i) {
        for (size_type j = 0; j < 4; ++j) {
            util::gtest_expect_floating_point_eq((*(params.data_ptr))[i][j], expected_libsvm[i][j], fmt::format("datapoint: {} feature: {}", i, j));
        }
    }

    params.parse_train_file(TEST_PATH "/data/5x4.arff");

    // check if sizes match
    ASSERT_EQ(params.data_ptr->size(), 5) << "num data points mismatch";
    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(params.data_ptr->operator[](i).size(), 4) << "mismatch num features in datapoint: " << i;
    }

    // correct values
    std::vector<std::vector<real_type>> expected_arff{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { 0., 0.60276937379453293, -0.13086851759108944, 0. },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };

    // check parsed values for correctness
    for (size_type i = 0; i < 5; ++i) {
        for (size_type j = 0; j < 4; ++j) {
            util::gtest_expect_floating_point_eq((*(params.data_ptr))[i][j], expected_arff[i][j], fmt::format("datapoint: {} feature: {}", i, j));
        }
    }
}

TYPED_TEST(BASE, parse_libsvm_ill_formed) {
    // setup parameters
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // parsing an arff file using the libsvm parser should result in an exception
    EXPECT_THROW(params.parse_libsvm(TEST_PATH "/data/5x4.arff", params.data_ptr), plssvm::invalid_file_format_exception);
}

TYPED_TEST(BASE, parse_arff_ill_formed) {
    // setup parameters
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // parsing a libsvm file using the arff parser should result in an exception
    EXPECT_THROW(params.parse_arff(TEST_PATH "/data/5x4.libsvm", params.data_ptr), plssvm::invalid_file_format_exception);
}

TYPED_TEST(BASE, parse_model_ill_formed) {
    // setup parameters
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // parse a libsvm file using the model file parser should result in an exception
    EXPECT_THROW(params.parse_model_file(TEST_PATH "/data/5x4.libsvm"), plssvm::invalid_file_format_exception);

    std::ifstream model_ifs(TEST_PATH "/data/models/5x4.libsvm.model");
    std::string correct_model((std::istreambuf_iterator<char>(model_ifs)), std::istreambuf_iterator<char>());

    const auto ill_formed_tester = [&params, correct_model](const std::string_view correct, const std::string_view altered, const std::string_view msg) {
        // alter correct model file to be ill-formed
        std::string ill_formed_model{ correct_model };
        plssvm::detail::replace_all(ill_formed_model, correct, altered);

        // create temporary file with ill-formed model specification
        std::string tmp_model_file = util::create_temp_file();
        std::ofstream ofs{ tmp_model_file };
        ofs << ill_formed_model;
        ofs.close();

        // perform actual check
        EXPECT_THROW_WHAT(params.parse_model_file(tmp_model_file), plssvm::invalid_file_format_exception, msg);

        // remove temporary file
        std::filesystem::remove(tmp_model_file);
    };

    // test svm_type
    ill_formed_tester("svm_type c_svc", "svm_type c_svc_wrong", "Can only use c_svc as svm_type, but 'c_svc_wrong' was given!");
    // test kernel
    ill_formed_tester("kernel_type linear", "kernel_type sigmoid", "Unrecognized kernel type 'sigmoid'!");
    // test number of classes
    ill_formed_tester("nr_class 2", "nr_class 3", "Can only use 2 classes, but 3 were given!");
    // test total number of support vectors
    ill_formed_tester("total_sv 5", "total_sv 0", "The number of support vectors must be greater than 0, but is 0!");
    // test labels
    ill_formed_tester("label 1 -1", "label 2 -1", "Only the labels 1 and -1 are allowed, but 'label 2 -1' were given!");
    ill_formed_tester("label 1 -1", "label 1 -2", "Only the labels 1 and -1 are allowed, but 'label 1 -2' were given!");
    ill_formed_tester("label 1 -1", "label 1 -1 2", "Only the labels 1 and -1 are allowed, but 'label 1 -1 2' were given!");
    ill_formed_tester("label 1 -1", "label 1", fmt::format("Can't convert '' to a value of type {}!", plssvm::detail::arithmetic_type_name<TypeParam>()));
    // test number of support vectors per class
    ill_formed_tester("nr_sv 2 3", "nr_sv 2 4", "The number of positive and negative support vectors doesn't add up to the total number: 2 + 4 != 5!");
    ill_formed_tester("nr_sv 2 3", "nr_sv 2 2 1", "Only two numbers are allowed, but more were given 'nr_sv 2 2 1'!");
    // test illegal entry
    ill_formed_tester("SV", "SV_wrong", "Unrecognized header entry 'SV_wrong'! Maybe SV is missing?");

    // test for missing total number of support vectors
    ill_formed_tester("total_sv 5\nnr_sv 2 3", "", "Missing total number of support vectors!");
    // test for missing labels
    ill_formed_tester("label 1 -1", "", "Missing labels!");
    // test for mussing number of support vectors per class
    ill_formed_tester("nr_sv 2 3", "", "Missing number of support vectors per class!");
    // test for missing rho
    ill_formed_tester("rho 0.37330625882191915", "", "Missing rho value!");
}

TYPED_TEST(BASE, parse_libsvm_non_existing_file) {
    // attempting to parse a non-existing file should result in an exception
    EXPECT_THROW(plssvm::parameter_train<TypeParam>{ TEST_PATH "/data/5x4.lib" }, plssvm::file_not_found_exception);
}

TYPED_TEST(BASE, parse_arff_non_existing_file) {
    // attempting to parse a non-existing file should result in an exception
    EXPECT_THROW(plssvm::parameter_train<TypeParam>{ TEST_PATH "/data/5x4.ar" }, plssvm::file_not_found_exception);
}

TYPED_TEST(BASE, parse_model_non_existing_file) {
    // attempting to parse a non-existing file should result in an exception
    EXPECT_THROW((plssvm::parameter_predict<TypeParam>{ TEST_PATH "/data/5x4.libsvm", TEST_PATH "/data/models/5x4.libsvm.mod" }), plssvm::file_not_found_exception);
    EXPECT_THROW((plssvm::parameter_predict<TypeParam>{ TEST_PATH "/data/5x4.lib", TEST_PATH "/data/models/5x4.libsvm.model" }), plssvm::file_not_found_exception);
}

TYPED_TEST(BASE, parameter_train) {
    plssvm::parameter<TypeParam> params;
    params.parse_train_file(TEST_PATH "/data/5x4.libsvm");

    plssvm::parameter_train<TypeParam> params_train{ TEST_PATH "/data/5x4.libsvm" };

    EXPECT_EQ(params.kernel, params_train.kernel);
    EXPECT_EQ(params.degree, params_train.degree);
    EXPECT_EQ(params.gamma, params_train.gamma);
    EXPECT_EQ(params.coef0, params_train.coef0);
    EXPECT_EQ(params.cost, params_train.cost);
    EXPECT_EQ(params.epsilon, params_train.epsilon);
    EXPECT_EQ(params.print_info, params_train.print_info);
    EXPECT_EQ(params.backend, params_train.backend);
    EXPECT_EQ(params.target, params_train.target);
    EXPECT_EQ(params.input_filename, params_train.input_filename);
    EXPECT_EQ(params.model_filename, params_train.model_filename);
    EXPECT_EQ(params.predict_filename, params_train.predict_filename);
    EXPECT_EQ(*params.data_ptr, *params_train.data_ptr);
    EXPECT_EQ(*params.value_ptr, *params_train.value_ptr);
    EXPECT_EQ(params.alphas_ptr, params_train.alphas_ptr);
    EXPECT_EQ(params.test_data_ptr, params_train.test_data_ptr);
}

TYPED_TEST(BASE, parameter_predict) {
    plssvm::parameter<TypeParam> params;
    params.parse_test_file(TEST_PATH "/data/5x4.libsvm");
    params.parse_model_file(TEST_PATH "/data/models/5x4.libsvm.model");

    plssvm::parameter_predict<TypeParam> params_predict{ TEST_PATH "/data/5x4.libsvm", TEST_PATH "/data/models/5x4.libsvm.model" };

    EXPECT_EQ(params.kernel, params_predict.kernel);
    EXPECT_EQ(params.degree, params_predict.degree);
    EXPECT_EQ(params.gamma, params_predict.gamma);
    EXPECT_EQ(params.coef0, params_predict.coef0);
    EXPECT_EQ(params.cost, params_predict.cost);
    EXPECT_EQ(params.epsilon, params_predict.epsilon);
    EXPECT_EQ(params.print_info, params_predict.print_info);
    EXPECT_EQ(params.backend, params_predict.backend);
    EXPECT_EQ(params.target, params_predict.target);
    EXPECT_EQ(params.input_filename, params_predict.input_filename);
    EXPECT_EQ(params.model_filename, params_predict.model_filename);
    EXPECT_EQ(params.predict_filename, params_predict.predict_filename);
    EXPECT_EQ(*params.data_ptr, *params_predict.data_ptr);
    EXPECT_EQ(*params.value_ptr, *params_predict.value_ptr);
    EXPECT_EQ(*params.alphas_ptr, *params_predict.alphas_ptr);
    EXPECT_EQ(*params.test_data_ptr, *params_predict.test_data_ptr);
}

TYPED_TEST(BASE, parameter_output) {
    // setup parameter and get output string
    plssvm::parameter<TypeParam> params;
    std::stringstream ss;
    ss << params;

    // correct output string
    std::string correct_output =
        fmt::format("kernel_type       linear\n"
                    "degree            3\n"
                    "gamma             0\n"
                    "coef0             0\n"
                    "cost              1\n"
                    "epsilon           0.001\n"
                    "print_info        true\n"
                    "backend           OpenMP\n"
                    "target platform   automatic\n"
                    "input_filename    ''\n"
                    "model_filename    ''\n"
                    "predict_filename  ''\n"
                    "rho               0\n"
                    "real_type         {}\n",
                    plssvm::detail::arithmetic_type_name<TypeParam>());

    // check for equality
    EXPECT_EQ(ss.str(), correct_output);
}

TYPED_TEST(BASE, transform_data) {
    // setup C-SVM
    plssvm::parameter_train<TypeParam> params{ TEST_PATH "/data/5x4.libsvm" };
    params.print_info = false;

    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    // transform data without and with boundary
    std::vector<real_type> result_no_boundary = csvm.transform_data(0);
    std::vector<real_type> result_boundary = csvm.transform_data(10);

    // check if sizes match
    EXPECT_EQ(result_no_boundary.size(), (csvm.get_num_data_points() - 1) * csvm.get_num_features());
    EXPECT_EQ(result_boundary.size(), (csvm.get_num_data_points() - 1 + 10) * csvm.get_num_features());

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
                fmt::format("datapoint: {} feature: {} at index: {}", datapoint, feature, datapoint + feature * (csvm.get_num_data_points() - 1 + 10)));
        }
    }
}

// enumerate all type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

template <typename T>
class BASE_parse_model : public ::testing::Test {};
TYPED_TEST_SUITE(BASE_parse_model, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(BASE_parse_model, parse_model_file) {
    // setup C-SVM class
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    switch (TypeParam::kernel) {
        case plssvm::kernel_type::linear:
            params.parse_model_file(TEST_PATH "/data/models/5x4.libsvm.model");
            break;
        case plssvm::kernel_type::polynomial:
            params.parse_model_file(TEST_PATH "/data/models/5x4.libsvm.polynomial.model");
            break;
        case plssvm::kernel_type::rbf:
            params.parse_model_file(TEST_PATH "/data/models/5x4.libsvm.rbf.model");
            break;
    }

    using real_type = typename decltype(params)::real_type;
    using size_type = typename decltype(params)::size_type;

    // check if necessary pointers are set
    ASSERT_NE(params.data_ptr, nullptr);
    ASSERT_NE(params.value_ptr, nullptr);
    ASSERT_NE(params.alphas_ptr, nullptr);

    // check if sizes match
    ASSERT_EQ(params.data_ptr->size(), 5) << "num support vectors mismatch";
    for (size_type i = 0; i < 5; ++i) {
        EXPECT_EQ(params.data_ptr->operator[](i).size(), 4) << "mismatch num features in support vector: " << i;
    }

    // correct support vectors
    std::vector<std::vector<real_type>> expected_model_support_vectors{
        { -1.117828, -2.908719, 0.6663834, 1.097883 },
        { -0.5282118, -0.335881, 0.5168730, 0.5460446 },
        { -0.2098121, 0.6027694, -0.1308685, 0.1080525 },
        { 1.884940, 1.005186, 0.2984999, 1.646463 },
        { 0.5765022, 1.014056, 0.1300943, 0.7261914 },
    };
    std::vector<real_type> expected_model_values{ 1, 1, -1, -1, -1 };
    std::vector<real_type> expected_model_alphas{ -0.17609610490769723, 0.8838187731213127, -0.47971257671001616, 0.0034556484621847128, -0.23146573996578407 };

    // check parsed values for correctness
    for (size_type i = 0; i < 5; ++i) {
        for (size_type j = 0; j < 4; ++j) {
            util::gtest_expect_floating_point_eq((*params.data_ptr)[i][j], expected_model_support_vectors[i][j], fmt::format("support vector: {} feature: {}", i, j));
        }
        EXPECT_EQ((*params.value_ptr)[i], expected_model_values[i]) << "support vector: " << i;
        util::gtest_expect_floating_point_eq((*params.alphas_ptr)[i], expected_model_alphas[i], fmt::format("support vector: {}", i));
    }

    // check if other parameter values are set correctly
    util::gtest_expect_floating_point_eq(params.rho, real_type{ 0.37330625882191915 });
    switch (TypeParam::kernel) {
        case plssvm::kernel_type::linear:
            EXPECT_EQ(params.kernel, plssvm::kernel_type::linear);
            break;
        case plssvm::kernel_type::polynomial:
            EXPECT_EQ(params.kernel, plssvm::kernel_type::polynomial);
            EXPECT_EQ(params.gamma, 0.25);
            break;
        case plssvm::kernel_type::rbf:
            EXPECT_EQ(params.kernel, plssvm::kernel_type::rbf);
            EXPECT_EQ(params.degree, 2);
            EXPECT_EQ(params.gamma, 0.25);
            EXPECT_EQ(params.coef0, 1.0);
            break;
    }
}

template <typename T>
class BASE_csvm :
    public ::testing::Test {};
TYPED_TEST_SUITE(BASE_csvm, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(BASE_csvm, csvm) {
    plssvm::parameter_train<typename TypeParam::real_type> params{ TEST_PATH "/data/5x4.libsvm" };  // TODO: change to params and check also parse model

    mock_csvm csvm{ params };

    EXPECT_EQ(params.kernel, csvm.kernel_);
    EXPECT_EQ(params.degree, csvm.get_degree());
    EXPECT_EQ(params.gamma, csvm.get_gamma());
    EXPECT_EQ(params.coef0, csvm.get_coef0());
    EXPECT_EQ(params.cost, csvm.get_cost());
    EXPECT_EQ(params.epsilon, csvm.epsilon_);
    EXPECT_EQ(params.print_info, csvm.print_info_);
    // EXPECT_EQ(params.print_info, csvm.get_alphas());
    // EXPECT_EQ(params.print_info, csvm.get_values());

    EXPECT_EQ((*(params.data_ptr)), csvm.get_data());
    EXPECT_EQ(params.data_ptr->size(), csvm.get_num_data_points());
    EXPECT_EQ((*(params.data_ptr))[0].size(), csvm.get_num_features());
}

template <typename T>
class BASE_write : public ::testing::Test {};
TYPED_TEST_SUITE(BASE_write, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(BASE_write, write_model) {
    // setup C-SVM
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.parse_train_file(TEST_PATH "/data/5x4.libsvm");
    params.kernel = TypeParam::kernel;

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
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type polynomial\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
        case plssvm::kernel_type::rbf:
            EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type rbf\ndegree [0-9]+\ngamma [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\ncoef0 [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nnr_class 2\ntotal_sv [0-9]+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));
            break;
        default:
            FAIL() << "untested kernel" << params.kernel;
            break;
    }
}

// generate tests for the kernel functions
template <typename T>
class BASE_kernel : public ::testing::Test {};
TYPED_TEST_SUITE(BASE_kernel, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(BASE_kernel, kernel_function) {
    // setup C-SVM
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    params.kernel = TypeParam::kernel;
    // set dummy data
    params.data_ptr = std::make_shared<const std::vector<std::vector<typename decltype(params)::real_type>>>(1);

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
    std::uniform_real_distribution<real_type> dist(-1, 2.0);
    std::generate(x1.begin(), x1.end(), [&]() { return dist(gen); });
    std::generate(x2.begin(), x2.end(), [&]() { return dist(gen); });

    // calculated result
    const real_type calculated = csvm.kernel_function(x1, x2);

    // correct result
    const real_type correct = compare::kernel_function<TypeParam::kernel>(x1, x2, csvm);

    // check for correctness
    util::gtest_expect_floating_point_eq(correct, calculated);
}