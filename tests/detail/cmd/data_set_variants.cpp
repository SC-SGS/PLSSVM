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

#include "plssvm/constants.hpp"                  // plssvm::real_type
#include "plssvm/detail/cmd/parser_predict.hpp"  // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"    // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"    // plssvm::detail::cmd::parser_train
#include "plssvm/mpi/communicator.hpp"           // plssvm::mpi::communicator
#include "plssvm/svm_types.hpp"                  // plssvm::svm_type

#include "tests/detail/cmd/cmd_utility.hpp"  // util::ParameterBase
#include "tests/naming.hpp"                  // naming::pretty_print_data_set_factory
#include "tests/utility.hpp"                 // util::{temporary_file, instantiate_template_file}

#include "fmt/format.h"   // fmt::format
#include "gtest/gtest.h"  // TEST_P, INSTANTIATE_TEST_SUITE_P, EXPECT_EQ,  ::testing::{WithParamInterface, Values}

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <tuple>    // std::tuple, std::make_tuple, std::ignore
#include <vector>   // std::vector

// the variant order is: classification<real_type, int> -> classification<real_type, std::string> -> regression<real_type, real_type>

class DataSetFactory : public util::ParameterBase,
                       public ::testing::WithParamInterface<std::tuple<bool, plssvm::svm_type, std::size_t>>,
                       protected util::temporary_file { };

TEST_P(DataSetFactory, data_set_factory_predict) {
    // get parameter
    const auto [strings_as_labels, svm, result_index] = GetParam();

    // assemble command line strings
    std::vector<std::string> cmd_args = { "./plssvm-predict" };
    if (strings_as_labels) {
        cmd_args.emplace_back("--use_strings_as_labels");
    }

    switch (svm) {
        case plssvm::svm_type::csvc:
            if (strings_as_labels) {
                util::instantiate_template_file<std::string>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
            } else {
                util::instantiate_template_file<int>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
            }
            cmd_args.insert(cmd_args.end(), { this->filename, PLSSVM_TEST_PATH "/data/model/classification/6x4.libsvm.model" });
            break;
        case plssvm::svm_type::csvr:
            util::instantiate_template_file<plssvm::real_type>(PLSSVM_TEST_PATH "/data/libsvm/regression/6x4.libsvm", this->filename);
            cmd_args.insert(cmd_args.end(), { this->filename, PLSSVM_TEST_PATH "/data/model/regression/6x4.libsvm.model" });
            break;
    }

    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(cmd_args);
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_comm(), this->get_argc(), this->get_argv() };

    // test active variant type
    const plssvm::detail::cmd::data_set_variants var = plssvm::detail::cmd::data_set_factory(this->get_comm(), parser);
    EXPECT_EQ(var.index(), result_index);
}

TEST_P(DataSetFactory, data_set_factory_scale) {
    // get parameter
    const auto [strings_as_labels, svm, result_index] = GetParam();

    if (strings_as_labels) {
        util::instantiate_template_file<std::string>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    } else {
        util::instantiate_template_file<int>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    }

    // assemble command line strings
    std::vector<std::string> cmd_args = { "./plssvm-scale" };
    if (strings_as_labels) {
        cmd_args.emplace_back("--use_strings_as_labels");
    }
    cmd_args.push_back(this->filename);

    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(cmd_args);
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->get_comm(), this->get_argc(), this->get_argv() };

    // test active variant type
    const plssvm::detail::cmd::data_set_variants var = plssvm::detail::cmd::data_set_factory(this->get_comm(), parser);
    if (svm == plssvm::svm_type::csvr) {
        // the svm_type doesn't matter for plssvm-scale
        EXPECT_EQ(var.index(), 0);  // use corresponding classification data set index
    } else {
        EXPECT_EQ(var.index(), result_index);
    }
}

TEST_P(DataSetFactory, data_set_factory_scale_restore_filename) {
    // get parameter
    const auto [strings_as_labels, svm, result_index] = GetParam();

    if (strings_as_labels) {
        util::instantiate_template_file<std::string>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    } else {
        util::instantiate_template_file<int>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    }

    // assemble command line strings
    std::vector<std::string> cmd_args = { "./plssvm-scale", "-r", PLSSVM_TEST_PATH "/data/scaling_factors/no_scaling_factors.txt" };
    if (strings_as_labels) {
        cmd_args.emplace_back("--use_strings_as_labels");
    }
    cmd_args.push_back(this->filename);

    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(cmd_args);
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->get_comm(), this->get_argc(), this->get_argv() };

    // test active variant type
    const plssvm::detail::cmd::data_set_variants var = plssvm::detail::cmd::data_set_factory(this->get_comm(), parser);
    if (svm == plssvm::svm_type::csvr) {
        // the svm_type doesn't matter for plssvm-scale
        EXPECT_EQ(var.index(), 0);  // use corresponding classification data set index
    } else {
        EXPECT_EQ(var.index(), result_index);
    }
}

TEST_P(DataSetFactory, data_set_factory_train) {
    // get parameter
    const auto [strings_as_labels, svm, result_index] = GetParam();

    if (strings_as_labels) {
        util::instantiate_template_file<std::string>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    } else {
        util::instantiate_template_file<int>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", this->filename);
    }

    // assemble command line strings
    std::vector<std::string> cmd_args = { "./plssvm-train", "--svm_type", fmt::format("{}", svm) };
    if (strings_as_labels) {
        cmd_args.emplace_back("--use_strings_as_labels");
    }
    cmd_args.push_back(this->filename);

    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(cmd_args);
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->get_comm(), this->get_argc(), this->get_argv() };

    // test active variant type
    const plssvm::detail::cmd::data_set_variants var = plssvm::detail::cmd::data_set_factory(this->get_comm(), parser);
    EXPECT_EQ(var.index(), result_index);
}

// clang-format off
// get<0>(tuple): whether the command line flag "string_as_labels" is provided (true) or not (false)
// get<1>(tuple): whether a C-SVC or C-SVR is used
// get<2>(tuple): the active index in the constructed variant
INSTANTIATE_TEST_SUITE_P(DataSetFactory, DataSetFactory, ::testing::Values(
                std::make_tuple(false, plssvm::svm_type::csvc, 0), std::make_tuple(true, plssvm::svm_type::csvc, 1), std::make_tuple(false, plssvm::svm_type::csvr, 2)),
                naming::pretty_print_data_set_factory<DataSetFactory>);
// clang-format on
