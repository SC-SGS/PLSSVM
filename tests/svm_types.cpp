/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the different SVM types.
 */

#include "plssvm/svm_types.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::Contains
#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_EQ

#include <sstream>      // std::istringstream
#include <string_view>  // std::string_view
#include <tuple>        // std::ignore
#include <vector>       // std::vector

// check whether the plssvm::svm_type -> std::string conversions are correct
TEST(SvmType, ToString) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::svm_type::csvc, "csvc");
    EXPECT_CONVERSION_TO_STRING(plssvm::svm_type::csvr, "csvr");
}

TEST(SvmType, ToStringUnknown) {
    // check conversions to std::string from unknown svm_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::svm_type>(2), "unknown");
}

// check whether the std::string -> plssvm::svm_type conversions are correct
TEST(SvmType, FromString) {
    // check conversion from std::string
    EXPECT_CONVERSION_FROM_STRING("CSVC", plssvm::svm_type::csvc);
    EXPECT_CONVERSION_FROM_STRING("csvc", plssvm::svm_type::csvc);
    EXPECT_CONVERSION_FROM_STRING("c-svc", plssvm::svm_type::csvc);
    EXPECT_CONVERSION_FROM_STRING("c_svc", plssvm::svm_type::csvc);
    EXPECT_CONVERSION_FROM_STRING("0", plssvm::svm_type::csvc);
    EXPECT_CONVERSION_FROM_STRING("CSVR", plssvm::svm_type::csvr);
    EXPECT_CONVERSION_FROM_STRING("csvr", plssvm::svm_type::csvr);
    EXPECT_CONVERSION_FROM_STRING("c-svr", plssvm::svm_type::csvr);
    EXPECT_CONVERSION_FROM_STRING("c_svr", plssvm::svm_type::csvr);
    EXPECT_CONVERSION_FROM_STRING("1", plssvm::svm_type::csvr);
}

TEST(SvmType, FromStringUnknown) {
    // foo isn't a valid svm_type
    std::istringstream input{ "foo" };
    plssvm::svm_type svm{};
    input >> svm;
    EXPECT_TRUE(input.fail());
}

TEST(SvmType, MinimalAvailableSvmTypes) {
    const std::vector<plssvm::svm_type> svms = plssvm::list_available_svm_types();

    // both C-SVM types must be available
    EXPECT_EQ(svms.size(), 2);
    EXPECT_THAT(svms, ::testing::Contains(plssvm::svm_type::csvc));
    EXPECT_THAT(svms, ::testing::Contains(plssvm::svm_type::csvr));
}

TEST(SvmType, SvmTypeToTaskName) {
    // get the task name from a C-SVC
    EXPECT_EQ(plssvm::svm_type_to_task_name(plssvm::svm_type::csvc), std::string_view{ "classification" });

    // get the task name from a C-SVR
    EXPECT_EQ(plssvm::svm_type_to_task_name(plssvm::svm_type::csvr), std::string_view{ "regression" });
}

TEST(SvmType, SvmTypeToTaskNameUnknown) {
    // try converting an unknown SVM type to a task name
    EXPECT_EQ(plssvm::svm_type_to_task_name(static_cast<plssvm::svm_type>(2)), std::string_view{ "unknown" });
}

TEST(SvmType, SvmTypeFromModelFile) {
    // check a classification model file
    EXPECT_EQ(plssvm::svm_type_from_model_file(PLSSVM_TEST_PATH "/data/model/classification/6x4.libsvm.model"), plssvm::svm_type::csvc);

    // check a regression model file
    EXPECT_EQ(plssvm::svm_type_from_model_file(PLSSVM_TEST_PATH "/data/model/regression/6x4.libsvm.model"), plssvm::svm_type::csvr);
}

TEST(SvmType, SvmTypeFromModelFileMissingSvmType) {
    // try getting the SVM type from an empty file won't work
    EXPECT_THROW_WHAT(std::ignore = plssvm::svm_type_from_model_file(PLSSVM_TEST_PATH "/data/model/classification/invalid/missing_svm_type.libsvm.model"),
                      plssvm::invalid_file_format_exception,
                      R"(The provided model file is not a valid LIBSVM model file since "svm_type" is missing!)");
}

TEST(SvmType, SvmTypeFromModelFileEmpty) {
    // try getting the SVM type from an empty file won't work
    EXPECT_THROW_WHAT(std::ignore = plssvm::svm_type_from_model_file(PLSSVM_TEST_PATH "/data/empty.txt"),
                      plssvm::invalid_file_format_exception,
                      R"(The provided model file is not a valid LIBSVM model file since "svm_type" AND "SV" are missing!)");
}
