/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the scale cmd parameter parsing.
 */

#include "plssvm/detail/cmd/parser_scale.hpp"

#include "plssvm/constants.hpp"  // plssvm::verbose

#include "../../naming.hpp"   // naming::{pretty_print_parameter_flag_and_value, pretty_print_parameter_flag}
#include "../../utility.hpp"  // util::{convert_to_string, convert_from_string}
#include "utility.hpp"        // util::ParameterBase

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::{StartsWith, HasSubstr}
#include "gtest/gtest.h"           // TEST_F, TEST_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EXIT, EXPECT_DEATH, INSTANTIATE_TEST_SUITE_P,
                                   // ::testing::WithParamInterface, ::testing::Combine, ::testing::Values, ::testing::Bool, ::testing::ExitedWithCode

#include <string>  // std::string
#include <tuple>   // std::tuple

class ParameterScale : public util::ParameterBase {};
class ParameterScaleDeathTest : public util::ParameterBase {};

TEST_F(ParameterScale, minimal) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale data.libsvm"));

    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };

    // check default values
    EXPECT_DOUBLE_EQ(-1.0, parser.lower);
    EXPECT_DOUBLE_EQ(+1.0, parser.upper);
    EXPECT_EQ(parser.format, plssvm::file_format_type::libsvm);
    EXPECT_FALSE(parser.strings_as_labels);
    EXPECT_FALSE(parser.float_as_real_type);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.scaled_filename, "data.libsvm.scaled");
    EXPECT_EQ(parser.save_filename, "");
    EXPECT_EQ(parser.restore_filename, "");
}
TEST_F(ParameterScale, minimal_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale data.libsvm"));

    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };

    // test output string
    const std::string correct =
        "lower: -1\n"
        "upper: 1\n"
        "label_type: int (default)\n"
        "real_type: double (default)\n"
        "output file format: libsvm\n"
        "input file: 'data.libsvm'\n"
        "scaled file: 'data.libsvm.scaled'\n"
        "save file (scaling factors): ''\n"
        "restore file (scaling factors): ''\n";
    EXPECT_EQ(util::convert_to_string(parser), correct);
}

TEST_F(ParameterScale, all_arguments) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale -l -2.0 -u 2.5 -f arff -s data.libsvm.save --use_strings_as_labels --use_float_as_real_type data.libsvm data.libsvm.scaled"));

    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };

    // check default values
    EXPECT_DOUBLE_EQ(-2.0, parser.lower);
    EXPECT_DOUBLE_EQ(+2.5, parser.upper);
    EXPECT_EQ(parser.format, plssvm::file_format_type::arff);
    EXPECT_TRUE(parser.strings_as_labels);
    EXPECT_TRUE(parser.float_as_real_type);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.scaled_filename, "data.libsvm.scaled");
    EXPECT_EQ(parser.save_filename, "data.libsvm.save");
    EXPECT_EQ(parser.restore_filename, "");
}
TEST_F(ParameterScale, all_arguments_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale -l -2.0 -u 2.5 -f arff -s data.libsvm.save --use_strings_as_labels --use_float_as_real_type data.libsvm data.libsvm.scaled"));

    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };

    // test output string
    const std::string correct =
        "lower: -2\n"
        "upper: 2.5\n"
        "label_type: std::string\n"
        "real_type: float\n"
        "output file format: arff\n"
        "input file: 'data.libsvm'\n"
        "scaled file: 'data.libsvm.scaled'\n"
        "save file (scaling factors): 'data.libsvm.save'\n"
        "restore file (scaling factors): ''\n";
    EXPECT_EQ(util::convert_to_string(parser), correct);
}

// test all command line parameter separately
class ParameterScaleLower : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParameterScaleLower, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.lower, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleLower, ::testing::Combine(
                ::testing::Values("-l", "--lower"),
                ::testing::Values(-2.5, -1.0, -0.01, 0.0)),
                naming::pretty_print_parameter_flag_and_value<ParameterScaleLower>);
// clang-format on

class ParameterScaleUpper : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParameterScaleUpper, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.upper, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleUpper,
                ::testing::Combine(::testing::Values("-u", "--upper"),
                ::testing::Values(0.0, 0.01, 1.0, 2.5)),
                naming::pretty_print_parameter_flag_and_value<ParameterScaleUpper>);
// clang-format on

class ParameterScaleFileFormat : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterScaleFileFormat, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to backend
    const auto backend = util::convert_from_string<plssvm::file_format_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.format, backend);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleFileFormat, ::testing::Combine(
                ::testing::Values("-f", "--format"),
                ::testing::Values("libsvm", "LIBSVM", "arff", "ARFF")),
                naming::pretty_print_parameter_flag_and_value<ParameterScaleFileFormat>);
// clang-format on

class ParameterScaleSaveFilename : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterScaleSaveFilename, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.save_filename, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleSaveFilename, ::testing::Combine(
                ::testing::Values("-s", "--save_filename"),
                ::testing::Values("data.libsvm.scaled", "output.txt")),
                naming::pretty_print_parameter_flag_and_value<ParameterScaleSaveFilename>);
// clang-format on

class ParameterScaleRestoreFilename : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterScaleRestoreFilename, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.restore_filename, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleRestoreFilename, ::testing::Combine(
                ::testing::Values("-r", "--restore_filename"),
                ::testing::Values("data.libsvm.weights", "output.txt")),
                naming::pretty_print_parameter_flag_and_value<ParameterScaleRestoreFilename>);
// clang-format on

class ParameterScaleUseStringsAsLabels : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, bool>> {};
TEST_P(ParameterScaleUseStringsAsLabels, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {}={} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.strings_as_labels, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleUseStringsAsLabels, ::testing::Combine(
                ::testing::Values("--use_strings_as_labels"),
                ::testing::Bool()),
                naming::pretty_print_parameter_flag_and_value<ParameterScaleUseStringsAsLabels>);
// clang-format on

class ParameterScaleUseFloatAsRealType : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, bool>> {};
TEST_P(ParameterScaleUseFloatAsRealType, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {}={} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.float_as_real_type, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleUseFloatAsRealType, ::testing::Combine(
                ::testing::Values("--use_float_as_real_type"),
                ::testing::Bool()),
                naming::pretty_print_parameter_flag_and_value<ParameterScaleUseFloatAsRealType>);
// clang-format on

class ParameterScaleQuiet : public ParameterScale, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterScaleQuiet, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} data.libsvm", flag));
    // create parameter object
    const plssvm::detail::cmd::parser_scale parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(plssvm::verbose, flag.empty());
}
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleQuiet, ::testing::Values("-q", "--quiet", ""), naming::pretty_print_parameter_flag<ParameterScaleQuiet>);

class ParameterScaleHelp : public ParameterScale, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterScaleHelp, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {}", flag));
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_scale{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleHelp, ::testing::Values("-h", "--help"), naming::pretty_print_parameter_flag<ParameterScaleHelp>);

class ParameterScaleVersion : public ParameterScale, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterScaleVersion, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {}", flag));
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_scale{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleVersion, ::testing::Values("-v", "--version"), naming::pretty_print_parameter_flag<ParameterScaleVersion>);

TEST_F(ParameterScaleDeathTest, no_positional_argument) {
    this->CreateCMDArgs("./plssvm-scale");
    EXPECT_EXIT((plssvm::detail::cmd::parser_scale{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::StartsWith("Error missing input file!"));
}
TEST_F(ParameterScaleDeathTest, save_and_restore) {
    this->CreateCMDArgs("./plssvm-scale -s data.libsvm.save -r data.libsvm.restore data.libsvm");
    EXPECT_EXIT((plssvm::detail::cmd::parser_scale{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::StartsWith("Error cannot use -s (--save_filename) and -r (--restore_filename) simultaneously!"));
}

TEST_F(ParameterScaleDeathTest, too_many_positional_arguments) {
    this->CreateCMDArgs("./plssvm-scale p1 p2 p3 p4");
    EXPECT_EXIT((plssvm::detail::cmd::parser_scale{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::HasSubstr(R"(Only up to two positional options may be given, but 2 ("p3 p4") additional option(s) where provided!)"));
}

// test whether nonsensical cmd arguments trigger the assertions
TEST_F(ParameterScaleDeathTest, too_few_argc) {
    EXPECT_DEATH((plssvm::detail::cmd::parser_scale{ 0, nullptr }),
                 ::testing::HasSubstr("At least one argument is always given (the executable name), but argc is 0!"));
}
TEST_F(ParameterScaleDeathTest, nullptr_argv) {
    EXPECT_DEATH((plssvm::detail::cmd::parser_scale{ 1, nullptr }),
                 ::testing::HasSubstr("At least one argument is always given (the executable name), but argv is a nullptr!"));
}
TEST_F(ParameterScaleDeathTest, illegal_scaling_range) {
    // illegal [lower, upper] bound range
    this->CreateCMDArgs("./plssvm-scale -l 1.0 -u -1.0 data.libsvm");
    EXPECT_DEATH((plssvm::detail::cmd::parser_scale{ this->argc, this->argv }),
                 ::testing::HasSubstr("Error invalid scaling range [lower, upper] with [1, -1]!"));
}
TEST_F(ParameterScaleDeathTest, unrecognized_option) {
    this->CreateCMDArgs("./plssvm-scale --foo bar");
    EXPECT_DEATH((plssvm::detail::cmd::parser_scale{ this->argc, this->argv }), "");
}