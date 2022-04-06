/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the parameter, parameter_train, and parameter_predict classes.
 */

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name
#include "plssvm/detail/string_conversion.hpp"     // plssvm::detail::convert_to
#include "plssvm/detail/string_utility.hpp"        // plssvm::detail::replace_all
#include "plssvm/exceptions/exceptions.hpp"        // plssvm::invalid_file_format_exception, plssvm::file_not_found_exception
#include "plssvm/kernel_types.hpp"                 // plssvm::kernel_type
#include "plssvm/parameter.hpp"                    // plssvm::parameter
#include "plssvm/parameter_predict.hpp"            // plssvm::parameter_predict
#include "plssvm/parameter_train.hpp"              // plssvm::parameter_train

#include "utility.hpp"    // util::gtest_expect_floating_point_eq, util::google_test::parameter_definition, util::google_test::parameter_definition_to_name,
                          // util::create_temp_file, EXPECT_THROW_WHAT
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads
#include "gtest/gtest.h"  // ::testing::Test, ::testing::Types, TYPED_TEST_SUITE, TYPED_TEST, ASSERT_EQ, ASSERT_NE,
                          // EXPECT_EQ, EXPECT_NE, EXPECT_TRUE, EXPECT_FALSE, EXPECT_THAT, EXPECT_THROW

#include <cstddef>           // std::size_t
#include <filesystem>        // std::filesystem::remove
#include <fstream>           // std::ifstream, std::ofstream
#include <initializer_list>  // std::initializer_list
#include <iterator>          // std::istreambuf_iterator
#include <memory>            // std::shared_ptr
#include <sstream>           // std::stringstream
#include <string>            // std::string
#include <string_view>       // std::string_view
#include <vector>            // std::vector

template <typename real_type, typename U>
std::vector<real_type> initialize_with_correct_type(const std::initializer_list<U> &data_vec) {
    if constexpr (std::is_same_v<real_type, U>) {
        return data_vec;
    } else {
        std::vector<real_type> correct_data_vec;
        correct_data_vec.reserve(data_vec.size());
        for (const U d : data_vec) {
            correct_data_vec.push_back(static_cast<real_type>(d));
        }
        return correct_data_vec;
    }
}

template <typename real_type, typename U>
std::vector<std::vector<real_type>> initialize_with_correct_type(const std::initializer_list<std::initializer_list<U>> &data) {
    std::vector<std::vector<real_type>> correct_data;
    for (const std::initializer_list<U> &data_vec : data) {
        correct_data.emplace_back(initialize_with_correct_type<real_type>(data_vec));
    }
    return correct_data;
}

// the floating point types to test
using floating_point_types = ::testing::Types<float, double>;

template <typename T>
class Parameter : public ::testing::Test {};
TYPED_TEST_SUITE(Parameter, floating_point_types);

// test whether plssvm::parameter<T>::parameter() correctly default initializes all member variables
TYPED_TEST(Parameter, default_constructor) {
    // create parameter object
    plssvm::parameter<TypeParam> params;

    using real_type = TypeParam;

    ::testing::StaticAssertTypeEq<real_type, typename decltype(params)::real_type>();

    EXPECT_EQ(params.kernel, plssvm::kernel_type::linear);
    EXPECT_EQ(params.degree, 3);
    EXPECT_EQ(params.gamma, real_type{ 0 });
    EXPECT_EQ(params.coef0, real_type{ 0 });
    EXPECT_EQ(params.cost, real_type{ 1 });
    EXPECT_EQ(params.epsilon, real_type{ 0.001 });
    EXPECT_TRUE(params.print_info);
    EXPECT_EQ(params.backend, plssvm::backend_type::openmp);
    EXPECT_EQ(params.target, plssvm::target_platform::automatic);

    EXPECT_TRUE(params.input_filename.empty());
    EXPECT_TRUE(params.model_filename.empty());
    EXPECT_TRUE(params.predict_filename.empty());

    EXPECT_EQ(params.data_ptr, nullptr);
    EXPECT_EQ(params.value_ptr, nullptr);
    EXPECT_EQ(params.alpha_ptr, nullptr);
    EXPECT_EQ(params.test_data_ptr, nullptr);

    EXPECT_EQ(params.rho, real_type{ 0 });
}

// utility function to reduce cody duplication
template <bool has_label, typename real_type>
void check_content_equal(const std::vector<std::vector<real_type>> &correct_data,
                         const std::vector<real_type> &correct_label,
                         const std::shared_ptr<const std::vector<std::vector<real_type>>> &parsed_data,
                         const std::shared_ptr<const std::vector<real_type>> &parsed_label) {
    ASSERT_EQ(correct_data.size(), correct_label.size());

    // check if sizes match
    ASSERT_NE(parsed_data, nullptr);
    ASSERT_EQ(parsed_data->size(), correct_data.size()) << "num data points mismatch";
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < parsed_data->size(); ++i) {
        ASSERT_EQ((*parsed_data)[i].size(), correct_data[i].size()) << "mismatch num features in data point: " << i;
    }
    if constexpr (has_label) {
        ASSERT_NE(parsed_label, nullptr);
        ASSERT_EQ(parsed_label->size(), correct_label.size()) << "num labels mismatch";
    } else {
        ASSERT_EQ(parsed_label, nullptr);
    }

    // check parsed values for correctness
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < parsed_data->size(); ++i) {
        for (typename std::vector<std::vector<real_type>>::size_type j = 0; j < (*parsed_data)[i].size(); ++j) {
            util::gtest_expect_floating_point_eq((*parsed_data)[i][j], correct_data[i][j], fmt::format("data point: {} feature: {}", i, j));
        }
        if constexpr (has_label) {
            EXPECT_EQ((*parsed_label)[i], correct_label[i]) << "data point: " << i;
        }
    }
}

//*************************************************************************************************************************************//
//                                                         parse libsvm files                                                          //
//*************************************************************************************************************************************//
// test whether plssvm::parameter<T>::parse_libsvm correctly parses libsvm files
TYPED_TEST(Parameter, parse_libsvm) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    using real_type = typename decltype(params)::real_type;

    // correct values
    const std::vector<std::vector<real_type>> expected_data = initialize_with_correct_type<real_type>({
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { -0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827 },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    });
    const std::vector<real_type> expected_values = initialize_with_correct_type<real_type>({ 1, 1, -1, -1, -1 });

    //
    // parse libsvm file with labels
    //
    params.parse_libsvm_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm", params.data_ptr);
    check_content_equal<true, real_type>(expected_data, expected_values, params.data_ptr, params.value_ptr);

    //
    // parse libsvm file without labels
    //
    params.parse_libsvm_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm.no_label", params.data_ptr);
    check_content_equal<false, real_type>(expected_data, expected_values, params.data_ptr, params.value_ptr);
}

// test whether plssvm::parameter<T>::parse_libsvm correctly parses sparse libsvm files
TYPED_TEST(Parameter, parse_libsvm_sparse) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    using real_type = typename decltype(params)::real_type;

    // correct values
    const std::vector<std::vector<real_type>> expected_data = initialize_with_correct_type<real_type>({
        { 0., 0., 0., 0. },
        { 0., 0., 0.51687296029754564, 0. },
        { 0., 1.01405596624706053, 0., 0. },
        { 0., 0.60276937379453293, 0., -0.13086851759108944 },
        { 0., 0., 0.298499933047586044, 0. },
    });
    const std::vector<real_type> expected_values = initialize_with_correct_type<real_type>({ 1, 1, -1, -1, -1 });

    //
    // parse libsvm file with labels
    //
    params.parse_libsvm_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.sparse.libsvm", params.data_ptr);
    check_content_equal<true, real_type>(expected_data, expected_values, params.data_ptr, params.value_ptr);

    //
    // parse libsvm file without labels
    //
    params.parse_libsvm_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.sparse.libsvm.no_label", params.data_ptr);
    check_content_equal<false, real_type>(expected_data, expected_values, params.data_ptr, params.value_ptr);
}

// test whether plssvm::parameter<T>::parse_libsvm correctly sets gamma
TYPED_TEST(Parameter, parse_libsvm_gamma) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // gamma = 1.0 (!= 0.0)
    params.gamma = 1.0;
    params.parse_libsvm_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm", params.data_ptr);

    using real_type = typename decltype(params)::real_type;

    util::gtest_assert_floating_point_eq(real_type{ 1.0 }, params.gamma);

    // gamma = 0.0 -> automatically set to (1.0 / num_features)
    params.gamma = 0.0;
    params.parse_libsvm_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm", params.data_ptr);

    ASSERT_NE(params.data_ptr, nullptr);
    ASSERT_GE(params.data_ptr->size(), 0);
    util::gtest_assert_floating_point_eq(real_type{ 1.0 } / static_cast<real_type>(params.data_ptr->back().size()), params.gamma);
}

// test whether plssvm::parameter<T>::parse_libsvm correctly fails parsing ill-formed libsvm files
TYPED_TEST(Parameter, parse_libsvm_ill_formed) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // parsing an arff file using the libsvm parser should result in an exception
    EXPECT_THROW(params.parse_libsvm_file(PLSSVM_TEST_PATH "/data/arff/5x4.arff", params.data_ptr), plssvm::invalid_file_format_exception);

    // test parsing an empty file
    EXPECT_THROW_WHAT(params.parse_libsvm_file(PLSSVM_TEST_PATH "/data/libsvm/0x0.libsvm", params.data_ptr), plssvm::invalid_file_format_exception, "Can't parse file: no data points are given!");
}

// test whether plssvm::parameter<T>::parse_libsvm correctly fails if the file doesn't exist
TYPED_TEST(Parameter, parse_libsvm_non_existing_file) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // attempting to parse a non-existing file should result in an exception
    EXPECT_THROW_WHAT(params.parse_libsvm_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.lib", params.data_ptr), plssvm::file_not_found_exception, fmt::format("Couldn't find file: '{}'!", PLSSVM_TEST_PATH "/data/libsvm/5x4.lib"));
}

//*************************************************************************************************************************************//
//                                                          parse arff files                                                           //
//*************************************************************************************************************************************//
// test whether plssvm::parameter<T>::parse_arff correctly parses arff files
TYPED_TEST(Parameter, parse_arff) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    using real_type = typename decltype(params)::real_type;

    // correct values
    const std::vector<std::vector<real_type>> expected_data = initialize_with_correct_type<real_type>({
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { 0., 0.60276937379453293, -0.13086851759108944, 0. },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    });
    const std::vector<real_type> expected_values = initialize_with_correct_type<real_type>({ 1, 1, -1, -1, -1 });

    //
    // parse arff file with labels
    //
    params.parse_arff_file(PLSSVM_TEST_PATH "/data/arff/5x4.arff", params.data_ptr);
    check_content_equal<true, real_type>(expected_data, expected_values, params.data_ptr, params.value_ptr);

    //
    // parse arff file without labels
    //
    params.parse_arff_file(PLSSVM_TEST_PATH "/data/arff/5x4.arff.no_label", params.data_ptr);
    check_content_equal<false, real_type>(expected_data, expected_values, params.data_ptr, params.value_ptr);
}

// test whether plssvm::parameter<T>::parse_arff correctly sets gamma
TYPED_TEST(Parameter, parse_arff_gamma) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // gamma = 1.0 (!= 0.0)
    params.gamma = 1.0;
    params.parse_arff_file(PLSSVM_TEST_PATH "/data/arff/5x4.arff", params.data_ptr);

    using real_type = typename decltype(params)::real_type;

    util::gtest_assert_floating_point_eq(real_type{ 1.0 }, params.gamma);

    // gamma = 0.0 -> automatically set to (1.0 / num_features)
    params.gamma = 0.0;
    params.parse_arff_file(PLSSVM_TEST_PATH "/data/arff/5x4.arff", params.data_ptr);

    ASSERT_NE(params.data_ptr, nullptr);
    ASSERT_GE(params.data_ptr->size(), 0);
    util::gtest_assert_floating_point_eq(real_type{ 1.0 } / static_cast<real_type>(params.data_ptr->back().size()), params.gamma);
}

// test whether plssvm::parameter<T>::parse_arff correctly fails parsing ill-formed arff files
TYPED_TEST(Parameter, parse_arff_ill_formed) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // parsing a libsvm file using the arff parser should result in an exception
    EXPECT_THROW(params.parse_arff_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm", params.data_ptr), plssvm::invalid_file_format_exception);

    // test parsing an empty file
    EXPECT_THROW_WHAT(params.parse_arff_file(PLSSVM_TEST_PATH "/data/libsvm/0x0.libsvm", params.data_ptr), plssvm::invalid_file_format_exception, "Can't parse file: no ATTRIBUTES are defined!");
    // test parsing a file without data points (but with @DATA)
    EXPECT_THROW_WHAT(params.parse_arff_file(PLSSVM_TEST_PATH "/data/arff/0x4.arff", params.data_ptr), plssvm::invalid_file_format_exception, "Can't parse file: no data points are given or @DATA is missing!");

    std::ifstream ifs(PLSSVM_TEST_PATH "/data/arff/5x4.arff");
    std::string correct_file((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());

    const auto ill_formed_tester = [&params, correct_file](const std::string_view correct, const std::string_view altered, const std::string_view msg) {
        // alter correct file to be ill-formed
        std::string ill_formed_model{ correct_file };
        plssvm::detail::replace_all(ill_formed_model, correct, altered);

        // create temporary file with ill-formed arff specification
        std::string tmp_model_file = util::create_temp_file();
        std::ofstream ofs{ tmp_model_file };
        ofs << ill_formed_model;
        ofs.close();

        // perform actual check
        EXPECT_THROW_WHAT(params.parse_arff_file(tmp_model_file, params.data_ptr), plssvm::invalid_file_format_exception, msg);

        // remove temporary file
        std::filesystem::remove(tmp_model_file);
    };

    // test for ATTRIBUTE with type not equal to NUMERIC
    ill_formed_tester("@ATTRIBUTE third    Numeric", "@ATTRIBUTE third    String", "Can only use NUMERIC features, but '@ATTRIBUTE third    String' was given!");
    // test for ATTRIBUTE with name class which isn't last
    ill_formed_tester("@ATTRIBUTE fourth   NUMERIC", "@ATTRIBUTE class    NUMERIC", "Only the last ATTRIBUTE may be CLASS!");
    // test for missing @DATA
    ill_formed_tester("@DATA", "", "Can't parse file: no data points are given or @DATA is missing!");

    // test for @ in data section
    ill_formed_tester("0.57650218263054642", "@0.57650218263054642", "Read @ inside data section!: '@0.57650218263054642,1.01405596624706053,0.13009428079760464,0.7261913886869387,-1'");
    // test for missing closing } in sparse data point
    ill_formed_tester("{1 0.60276937379453293, 2 -0.13086851759108944, 4 -1}", "{1 0.60276937379453293, 2 -0.13086851759108944, 4 -1", "Missing closing '}' for sparse data point 3 description!");
    // test for missing label even though @ATTRIBUTE class is defined
    ill_formed_tester("{1 0.60276937379453293, 2 -0.13086851759108944, 4 -1}", "{1 0.60276937379453293, 2 -0.13086851759108944}", "Missing label for data point 3!");
    // test for missing data point
    ill_formed_tester("-1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288,1", "-1.117827500607882,-2.9087188881250993,1", "Invalid number of features/labels! Found 2 but should be 4!");
    // test for additional data point
    ill_formed_tester("-1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288,1", "-1.117827500607882,-2.9087188881250993,1.0978832703949288,1.0978832703949288,1.0978832703949288,1", "Too many features! Superfluous ',1' for data point 0!");
    // test for additional data point in sparse specification
    ill_formed_tester("{1 0.60276937379453293, 2 -0.13086851759108944, 4 -1}", "{1 0.60276937379453293, 2 -0.13086851759108944, 4 -1, 5 42.1415}", "Too many features given! Trying to add feature at position 5 but max position is 3!");
    // test for missing label
    ill_formed_tester("-1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288,1", "-1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288", "Invalid number of features/labels! Found 3 but should be 4!");
}

// test whether plssvm::parameter<T>::parse_arff correctly fails if the file doesn't exist
TYPED_TEST(Parameter, parse_arff_non_existing_file) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    // attempting to parse a non-existing file should result in an exception
    EXPECT_THROW_WHAT(params.parse_arff_file(PLSSVM_TEST_PATH "/data/arff/5x4.ar", params.data_ptr), plssvm::file_not_found_exception, fmt::format("Couldn't find file: '{}'!", PLSSVM_TEST_PATH "/data/arff/5x4.ar"));
}

// test whether plssvm::parameter<T>::parse_file uses the correct file parser
TYPED_TEST(Parameter, parse_file) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    using real_type = typename decltype(params)::real_type;

    // correct values for libsvm file
    const std::vector<std::vector<real_type>> expected_data_libsvm = initialize_with_correct_type<real_type>({
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { -0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827 },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    });
    const std::vector<real_type> expected_values_libsvm = initialize_with_correct_type<real_type>({ 1, 1, -1, -1, -1 });

    //
    // parse libsvm file
    //
    params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");
    check_content_equal<true, real_type>(expected_data_libsvm, expected_values_libsvm, params.data_ptr, params.value_ptr);

    // correct values for arff file
    const std::vector<std::vector<real_type>> expected_data_arff = initialize_with_correct_type<real_type>({
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { 0., 0.60276937379453293, -0.13086851759108944, 0. },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    });
    const std::vector<real_type> expected_values_arff = initialize_with_correct_type<real_type>({ 1, 1, -1, -1, -1 });

    //
    // parse arff file
    //
    params.parse_train_file(PLSSVM_TEST_PATH "/data/arff/5x4.arff");
    check_content_equal<true, real_type>(expected_data_arff, expected_values_arff, params.data_ptr, params.value_ptr);
}

// test whether plssvm::parameter<T>::parse_train_file uses the correct data pointer
TYPED_TEST(Parameter, parse_train_file) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    using real_type = typename decltype(params)::real_type;

    // correct values
    const std::vector<std::vector<real_type>> expected_data = initialize_with_correct_type<real_type>({
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { -0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827 },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    });
    const std::vector<real_type> expected_values = initialize_with_correct_type<real_type>({ 1, 1, -1, -1, -1 });

    //
    // parse train file with labels
    //
    params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");
    check_content_equal<true, real_type>(expected_data, expected_values, params.data_ptr, params.value_ptr);

    //
    // parse train file without labels should result in an exception
    //
    EXPECT_THROW_WHAT(params.parse_train_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm.no_label"), plssvm::invalid_file_format_exception, "Missing labels for train file!");
}

// test whether plssvm::parameter<T>::parse_test_file uses the correct test data pointer
TYPED_TEST(Parameter, parse_test_file) {
    // create parameter object
    plssvm::parameter<TypeParam> params;
    params.print_info = false;

    using real_type = typename decltype(params)::real_type;

    // correct values
    const std::vector<std::vector<real_type>> expected_data = initialize_with_correct_type<real_type>({
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { -0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827 },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    });
    const std::vector<real_type> expected_values = initialize_with_correct_type<real_type>({ 1, 1, -1, -1, -1 });

    //
    // parse test file with labels
    //
    params.parse_test_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");
    check_content_equal<true, real_type>(expected_data, expected_values, params.test_data_ptr, params.value_ptr);

    //
    // parse test file without labels
    //
    params.parse_test_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm.no_label");
    check_content_equal<false, real_type>(expected_data, expected_values, params.test_data_ptr, params.value_ptr);
}

//*************************************************************************************************************************************//
//                                                         parse model files                                                           //
//*************************************************************************************************************************************//

// enumerate all floating point type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<float, plssvm::kernel_type::rbf>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::polynomial>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::rbf>>;

template <typename T>
class ParameterModel : public ::testing::Test {};
TYPED_TEST_SUITE(ParameterModel, parameter_types, util::google_test::parameter_definition_to_name);

// test whether plssvm::parameter<T>::parse_model_file correctly parses model files
TYPED_TEST(ParameterModel, parse_model_file) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;
    // parse model file based on used kernel type
    switch (TypeParam::kernel) {
        case plssvm::kernel_type::linear:
            params.parse_model_file(PLSSVM_TEST_PATH "/data/models/5x4.libsvm.model");
            break;
        case plssvm::kernel_type::polynomial:
            params.parse_model_file(PLSSVM_TEST_PATH "/data/models/5x4.libsvm.polynomial.model");
            break;
        case plssvm::kernel_type::rbf:
            params.parse_model_file(PLSSVM_TEST_PATH "/data/models/5x4.libsvm.rbf.model");
            break;
    }

    using real_type = typename decltype(params)::real_type;

    // correct support vectors
    const std::vector<std::vector<real_type>> expected_model_support_vectors = initialize_with_correct_type<real_type>({
        { -1.117828, -2.908719, 0.6663834, 1.097883 },
        { -0.5282118, -0.335881, 0.5168730, 0.5460446 },
        { -0.2098121, 0.6027694, -0.1308685, 0.1080525 },
        { 1.884940, 1.005186, 0.2984999, 1.646463 },
        { 0.5765022, 1.014056, 0.1300943, 0.7261914 },
    });
    const std::vector<real_type> expected_model_alphas = initialize_with_correct_type<real_type>({ -0.17609610490769723, 0.8838187731213127, -0.47971257671001616, 0.0034556484621847128, -0.23146573996578407 });

    // check if necessary pointers are set
    ASSERT_NE(params.value_ptr, nullptr);

    // check if sizes match
    ASSERT_NE(params.data_ptr, nullptr);
    ASSERT_EQ(params.data_ptr->size(), expected_model_support_vectors.size()) << "num support vectors mismatch";
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < params.data_ptr->size(); ++i) {
        EXPECT_EQ((*params.data_ptr)[i].size(), expected_model_support_vectors[i].size()) << "mismatch num features in support vector: " << i;
    }
    ASSERT_NE(params.alpha_ptr, nullptr);
    ASSERT_EQ(params.alpha_ptr->size(), expected_model_alphas.size()) << "num alphas mismatch";

    // check parsed values for correctness
    for (typename std::vector<std::vector<real_type>>::size_type i = 0; i < params.data_ptr->size(); ++i) {
        for (typename std::vector<std::vector<real_type>>::size_type j = 0; j < (*params.data_ptr)[i].size(); ++j) {
            util::gtest_expect_floating_point_eq((*params.data_ptr)[i][j], expected_model_support_vectors[i][j], fmt::format("support vector: {} feature: {}", i, j));
        }
        util::gtest_expect_floating_point_eq((*params.alpha_ptr)[i], expected_model_alphas[i], fmt::format("support vector: {}", i));
    }

    // check if other parameter values are set correctly
    util::gtest_expect_floating_point_eq(params.rho, plssvm::detail::convert_to<real_type>("0.37330625882191915"));
    switch (TypeParam::kernel) {
        case plssvm::kernel_type::linear:
            EXPECT_EQ(params.kernel, plssvm::kernel_type::linear);
            break;
        case plssvm::kernel_type::polynomial:
            EXPECT_EQ(params.kernel, plssvm::kernel_type::polynomial);
            EXPECT_EQ(params.degree, 2);
            EXPECT_EQ(params.gamma, real_type{ 0.25 });
            EXPECT_EQ(params.coef0, real_type{ 1 });
            break;
        case plssvm::kernel_type::rbf:
            EXPECT_EQ(params.kernel, plssvm::kernel_type::rbf);
            EXPECT_EQ(params.gamma, real_type{ 0.25 });
            break;
    }
}

// test whether plssvm::parameter<T>::parse_model_file correctly fails parsing ill-formed model files
TYPED_TEST(ParameterModel, parse_model_ill_formed) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;

    using real_type = typename decltype(params)::real_type;

    // parse a libsvm file using the model file parser should result in an exception
    EXPECT_THROW(params.parse_model_file(PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm"), plssvm::invalid_file_format_exception);
    // test parsing a file without data points (but with SV)
    EXPECT_THROW_WHAT(params.parse_model_file(PLSSVM_TEST_PATH "/data/models/0x4.model"), plssvm::invalid_file_format_exception, "Can't parse file: no support vectors are given or SV is missing!");

    std::ifstream model_ifs(PLSSVM_TEST_PATH "/data/models/5x4.libsvm.model");
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
    ill_formed_tester("label 1 -1", "label 1", fmt::format("Can't convert '' to a value of type {}!", plssvm::detail::arithmetic_type_name<real_type>()));
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

// test whether plssvm::parameter<T>::parse_model_file correctly fails if the file doesn't exist
TYPED_TEST(ParameterModel, parse_model_non_existing_file) {
    // create parameter object
    plssvm::parameter<typename TypeParam::real_type> params;
    params.print_info = false;

    // attempting to parse a non-existing file should result in an exception
    EXPECT_THROW_WHAT(params.parse_model_file(PLSSVM_TEST_PATH "/data/models/5x4.libsvm.mod"), plssvm::file_not_found_exception, fmt::format("Couldn't find file: '{}'!", PLSSVM_TEST_PATH "/data/models/5x4.libsvm.mod"));
}

//*************************************************************************************************************************************//
//                                                         operator>> overload                                                         //
//*************************************************************************************************************************************//
// check whether the operator<<(plssvm::parameter<T>) output is correct
TYPED_TEST(Parameter, output_operator) {
    // create parameter object
    plssvm::parameter<TypeParam> params;

    // get output string
    std::stringstream ss;
    ss << params;

    // correct output string
    std::string correct_output =
        fmt::format("kernel_type                 linear\n"
                    "degree                      3\n"
                    "gamma                       0\n"
                    "coef0                       0\n"
                    "cost                        1\n"
                    "epsilon                     0.001\n"
                    "print_info                  true\n"
                    "backend                     openmp\n"
                    "target platform             automatic\n"
                    "SYCL kernel invocation type automatic\n"
                    "SYCL implementation type    automatic\n"
                    "input_filename              ''\n"
                    "model_filename              ''\n"
                    "predict_filename            ''\n"
                    "rho                         0\n"
                    "real_type                   {}\n",
                    plssvm::detail::arithmetic_type_name<TypeParam>());

    // check for equality
    EXPECT_EQ(ss.str(), correct_output);
}

//*************************************************************************************************************************************//
//                                                           parameter train                                                           //
//*************************************************************************************************************************************//
template <typename T>
class ParameterTrain : public ::testing::Test {};
TYPED_TEST_SUITE(ParameterTrain, floating_point_types);
// check whether the single std::string constructor of plssvm::parameter_train<T> is correct
TYPED_TEST(ParameterTrain, parse_filename) {
    // create parameter object
    plssvm::parameter_train<TypeParam> params{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };

    using real_type = TypeParam;

    ::testing::StaticAssertTypeEq<real_type, typename decltype(params)::real_type>();

    EXPECT_EQ(params.kernel, plssvm::kernel_type::linear);
    EXPECT_EQ(params.degree, 3);
    EXPECT_EQ(params.gamma, real_type{ 0.25 });
    EXPECT_EQ(params.coef0, real_type{ 0 });
    EXPECT_EQ(params.cost, real_type{ 1 });
    EXPECT_EQ(params.epsilon, real_type{ 0.001 });
    EXPECT_TRUE(params.print_info);
    EXPECT_EQ(params.backend, plssvm::backend_type::openmp);
    EXPECT_EQ(params.target, plssvm::target_platform::automatic);

    EXPECT_EQ(params.input_filename, PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");
    EXPECT_EQ(params.model_filename, "5x4.libsvm.model");
    EXPECT_TRUE(params.predict_filename.empty());

    EXPECT_NE(params.data_ptr, nullptr);
    EXPECT_NE(params.value_ptr, nullptr);
    EXPECT_EQ(params.alpha_ptr, nullptr);
    EXPECT_EQ(params.test_data_ptr, nullptr);

    EXPECT_EQ(params.rho, real_type{ 0 });
}

// check whether the command line parsing for plssvm::parameter_train<T> is correct
TYPED_TEST(ParameterTrain, parse_command_line_arguments) {
    // used command line parameters
    std::vector<std::string> argv_vec = { "./plssvm-train", "--backend", "cuda", "--target_platform", "gpu_nvidia", "-t", "1", "-d", "5", "--gamma", "3.1415", "-r", "0.42", "--cost", "1.89", "-e", "0.00001", "-q", "--input", PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm" };
    std::vector<char *> argv(argv_vec.size());
    for (std::size_t i = 0; i < argv.size(); ++i) {
        argv[i] = argv_vec[i].data();
    }

    // create parameter object
    plssvm::parameter_train<TypeParam> params(static_cast<int>(argv.size()), argv.data());

    using real_type = TypeParam;

    ::testing::StaticAssertTypeEq<real_type, typename decltype(params)::real_type>();

    EXPECT_EQ(params.kernel, plssvm::kernel_type::polynomial);
    EXPECT_EQ(params.degree, 5);
    EXPECT_EQ(params.gamma, real_type{ 3.1415 });
    EXPECT_EQ(params.coef0, real_type{ 0.42 });
    EXPECT_EQ(params.cost, real_type{ 1.89 });
    EXPECT_EQ(params.epsilon, real_type{ 0.00001 });
    EXPECT_FALSE(params.print_info);
    EXPECT_EQ(params.backend, plssvm::backend_type::cuda);
    EXPECT_EQ(params.target, plssvm::target_platform::gpu_nvidia);

    EXPECT_EQ(params.input_filename, PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");
    EXPECT_EQ(params.model_filename, "5x4.libsvm.model");
    EXPECT_TRUE(params.predict_filename.empty());

    EXPECT_NE(params.data_ptr, nullptr);
    EXPECT_NE(params.value_ptr, nullptr);
    EXPECT_EQ(params.alpha_ptr, nullptr);
    EXPECT_EQ(params.test_data_ptr, nullptr);

    EXPECT_EQ(params.rho, real_type{ 0 });
}

//*************************************************************************************************************************************//
//                                                          parameter predict                                                          //
//*************************************************************************************************************************************//
template <typename T>
class ParameterPredict : public ::testing::Test {};
TYPED_TEST_SUITE(ParameterPredict, floating_point_types);
// check whether the std::string constructor of plssvm::parameter_predict<T> is correct
TYPED_TEST(ParameterPredict, parse_filename) {
    // create parameter object
    plssvm::parameter_predict<TypeParam> params{ PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm", PLSSVM_TEST_PATH "/data/models/5x4.libsvm.model" };

    using real_type = TypeParam;

    ::testing::StaticAssertTypeEq<real_type, typename decltype(params)::real_type>();

    EXPECT_EQ(params.kernel, plssvm::kernel_type::linear);
    EXPECT_EQ(params.degree, 3);
    EXPECT_EQ(params.gamma, real_type{ 0.25 });
    EXPECT_EQ(params.coef0, real_type{ 0 });
    EXPECT_EQ(params.cost, real_type{ 1 });
    EXPECT_EQ(params.epsilon, real_type{ 0.001 });
    EXPECT_TRUE(params.print_info);
    EXPECT_EQ(params.backend, plssvm::backend_type::openmp);
    EXPECT_EQ(params.target, plssvm::target_platform::automatic);

    EXPECT_EQ(params.input_filename, PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");
    EXPECT_EQ(params.model_filename, PLSSVM_TEST_PATH "/data/models/5x4.libsvm.model");
    EXPECT_EQ(params.predict_filename, "5x4.libsvm.predict");

    EXPECT_NE(params.data_ptr, nullptr);
    EXPECT_NE(params.value_ptr, nullptr);
    EXPECT_NE(params.alpha_ptr, nullptr);
    EXPECT_NE(params.test_data_ptr, nullptr);

    util::gtest_expect_floating_point_eq(params.rho, plssvm::detail::convert_to<real_type>("0.37330625882191915"));
}

// check whether the command line parsing for plssvm::parameter_predict<T> is correct
TYPED_TEST(ParameterPredict, parse_command_line_arguments) {
    // used command line parameters

    std::vector<std::string> argv_vec = { "./plssvm-predict", "--backend", "opencl", "--target_platform", "cpu", "--test", PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm", "--model", PLSSVM_TEST_PATH "/data/models/5x4.libsvm.model" };
    std::vector<char *> argv(argv_vec.size());
    for (std::size_t i = 0; i < argv.size(); ++i) {
        argv[i] = argv_vec[i].data();
    }

    // create parameter object
    plssvm::parameter_predict<TypeParam> params(static_cast<int>(argv.size()), argv.data());

    using real_type = TypeParam;

    ::testing::StaticAssertTypeEq<real_type, typename decltype(params)::real_type>();

    EXPECT_EQ(params.kernel, plssvm::kernel_type::linear);
    EXPECT_EQ(params.degree, 3);
    EXPECT_EQ(params.gamma, real_type{ 0.25 });
    EXPECT_EQ(params.coef0, real_type{ 0 });
    EXPECT_EQ(params.cost, real_type{ 1 });
    EXPECT_EQ(params.epsilon, real_type{ 0.001 });
    EXPECT_TRUE(params.print_info);
    EXPECT_EQ(params.backend, plssvm::backend_type::opencl);
    EXPECT_EQ(params.target, plssvm::target_platform::cpu);

    EXPECT_EQ(params.input_filename, PLSSVM_TEST_PATH "/data/libsvm/5x4.libsvm");
    EXPECT_EQ(params.model_filename, PLSSVM_TEST_PATH "/data/models/5x4.libsvm.model");
    EXPECT_EQ(params.predict_filename, "5x4.libsvm.predict");

    EXPECT_NE(params.data_ptr, nullptr);
    EXPECT_NE(params.value_ptr, nullptr);
    EXPECT_NE(params.alpha_ptr, nullptr);
    EXPECT_NE(params.test_data_ptr, nullptr);

    util::gtest_expect_floating_point_eq(params.rho, plssvm::detail::convert_to<real_type>("0.37330625882191915"));
}