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

#include "tests/custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_CONVERSION_FROM_STRING

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::Contains
#include "gtest/gtest.h"  // TEST, EXPECT_TRUE, EXPECT_EQ

#include <sstream>      // std::istringstream
#include <string_view>  // std::string_view
#include <vector>       // std::vector

// check whether the plssvm::svm_type -> std::string conversions are correct
TEST(SvmType, to_string) {
    // check conversions to std::string
    EXPECT_CONVERSION_TO_STRING(plssvm::svm_type::csvc, "csvc");
    EXPECT_CONVERSION_TO_STRING(plssvm::svm_type::csvr, "csvr");
}

TEST(SvmType, to_string_unknown) {
    // check conversions to std::string from unknown svm_type
    EXPECT_CONVERSION_TO_STRING(static_cast<plssvm::svm_type>(2), "unknown");
}

// check whether the std::string -> plssvm::svm_type conversions are correct
TEST(SvmType, from_string) {
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

TEST(SvmType, from_string_unknown) {
    // foo isn't a valid svm_type
    std::istringstream input{ "foo" };
    plssvm::svm_type svm{};
    input >> svm;
    EXPECT_TRUE(input.fail());
}

TEST(SvmType, minimal_available_svm_types) {
    const std::vector<plssvm::svm_type> svms = plssvm::list_available_svm_types();

    // both CSVM types must be available
    EXPECT_EQ(svms.size(), 2);
    EXPECT_THAT(svms, ::testing::Contains(plssvm::svm_type::csvc));
    EXPECT_THAT(svms, ::testing::Contains(plssvm::svm_type::csvr));
}

TEST(SvmType, svm_type_to_task_name) {
    // get the task name from a CSVC
    EXPECT_EQ(plssvm::svm_type_to_task_name(plssvm::svm_type::csvc), std::string_view{ "classification" });

    // get the task name from a CSVR
    EXPECT_EQ(plssvm::svm_type_to_task_name(plssvm::svm_type::csvr), std::string_view{ "regression" });
}

TEST(SvmType, svm_type_from_model_file) {
    // check a classification model file
    EXPECT_EQ(plssvm::svm_type_from_model_file(PLSSVM_TEST_PATH "/data/model/6x4_linear.libsvm.model"), plssvm::svm_type::csvc);

    // TODO: CSVR model file
    // check a regression model file
    // EXPECT_EQ(plssvm::svm_type_from_model_file(PLSSVM_TEST_PATH "/data/model/6x4_linear.libsvm.model"), plssvm::svm_type::csvr);
}
