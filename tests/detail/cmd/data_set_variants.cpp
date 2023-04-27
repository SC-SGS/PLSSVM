/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the data variant creation based on the provided runtime parameters.
 */

#include "plssvm/detail/cmd/data_set_variants.hpp"

#include "plssvm/detail/cmd/parser_predict.hpp"  // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"    // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"    // plssvm::detail::cmd::parser_train

#include "../../naming.hpp"                      // naming::pretty_print_data_set_factory
#include "../../utility.hpp"                     // util::{temporary_file, instantiate_template_file}
#include "utility.hpp"                           // util::ParameterBase

#include "fmt/core.h"                            // fmt::format
#include "gtest/gtest.h"                         // TEST_P, INSTANTIATE_TEST_SUITE_P, EXPECT_EQ,  ::testing::{WithParamInterface, Values}

#include <cstddef>                               // std::size_t
#include <string>                                // std::string
#include <tuple>                                 // std::tuple, std::make_tuple

// the variant order is: <float, int>, <float, std::string>, <double, int>, <double, std::string>

class DataSetFactory : public util::ParameterBase, public ::testing::WithParamInterface<std::tuple<bool, bool, std::size_t>>, protected util::temporary_file {};

TEST_P(DataSetFactory, data_set_factory_predict) {
    // get parameter
    const auto [float_as_real_type, strings_as_labels, index] = GetParam();

    if (strings_as_labels) {
        util::instantiate_template_file<std::string>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    } else {
        util::instantiate_template_file<int>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    }

    // assemble command line strings
    std::vector<std::string> cmd_args = { "./plssvm-predict" };
    if (strings_as_labels) {
        cmd_args.emplace_back("--use_strings_as_labels");
    }
    if (float_as_real_type) {
        cmd_args.emplace_back("--use_float_as_real_type");
    }
    cmd_args.insert(cmd_args.end(), { this->filename, "data.libsvm.model" });

    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(cmd_args);
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };

    // test active variant type
    const plssvm::detail::cmd::data_set_variants var = plssvm::detail::cmd::data_set_factory(parser);
    EXPECT_EQ(var.index(), index);
}
TEST_P(DataSetFactory, data_set_factory_scale) {
    // get parameter
    const auto [float_as_real_type, strings_as_labels, index] = GetParam();

    if (strings_as_labels) {
        util::instantiate_template_file<std::string>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    } else {
        util::instantiate_template_file<int>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    }

    // assemble command line strings
    std::vector<std::string> cmd_args = { "./plssvm-scale" };
    if (strings_as_labels) {
        cmd_args.emplace_back("--use_strings_as_labels");
    }
    if (float_as_real_type) {
        cmd_args.emplace_back("--use_float_as_real_type");
    }
    cmd_args.push_back(this->filename);

    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(cmd_args);
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };

    // test active variant type
    const plssvm::detail::cmd::data_set_variants var = plssvm::detail::cmd::data_set_factory(parser);
    EXPECT_EQ(var.index(), index);
}
TEST_P(DataSetFactory, data_set_factory_scale_restore_filename) {
    // get parameter
    const auto [float_as_real_type, strings_as_labels, index] = GetParam();

    if (strings_as_labels) {
        util::instantiate_template_file<std::string>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    } else {
        util::instantiate_template_file<int>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    }

    // assemble command line strings
    std::vector<std::string> cmd_args = { "./plssvm-scale", "-r", PLSSVM_TEST_PATH "/data/scaling_factors/no_scaling_factors.txt" };
    if (strings_as_labels) {
        cmd_args.emplace_back("--use_strings_as_labels");
    }
    if (float_as_real_type) {
        cmd_args.emplace_back("--use_float_as_real_type");
    }
    cmd_args.push_back(this->filename);

    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(cmd_args);
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };

    // test active variant type
    const plssvm::detail::cmd::data_set_variants var = plssvm::detail::cmd::data_set_factory(parser);
    EXPECT_EQ(var.index(), index);
}
TEST_P(DataSetFactory, data_set_factory_train) {
    // get parameter
    const auto [float_as_real_type, strings_as_labels, index] = GetParam();

    if (strings_as_labels) {
        util::instantiate_template_file<std::string>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    } else {
        util::instantiate_template_file<int>(PLSSVM_TEST_PATH "/data/libsvm/5x4_TEMPLATE.libsvm", this->filename);
    }

    // assemble command line strings
    std::vector<std::string> cmd_args = { "./plssvm-train" };
    if (strings_as_labels) {
        cmd_args.emplace_back("--use_strings_as_labels");
    }
    if (float_as_real_type) {
        cmd_args.emplace_back("--use_float_as_real_type");
    }
    cmd_args.push_back(this->filename);

    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(cmd_args);
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };

    // test active variant type
    const plssvm::detail::cmd::data_set_variants var = plssvm::detail::cmd::data_set_factory(parser);
    EXPECT_EQ(var.index(), index);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(DataSetFactory, DataSetFactory, ::testing::Values(
                std::make_tuple(false, false, 2), std::make_tuple(false, true, 3),
                std::make_tuple(true, false, 0), std::make_tuple(true, true, 1)),
                naming::pretty_print_data_set_factory<DataSetFactory>);
// clang-format on