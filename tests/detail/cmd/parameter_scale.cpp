/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the scale cmd parameter parsing.
 */

#include "plssvm/detail/cmd/parameter_scale.hpp"
#include "plssvm/constants.hpp"  // plssvm::verbose

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
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };

    // check default values
    EXPECT_DOUBLE_EQ(-1.0, params.lower);
    EXPECT_DOUBLE_EQ(+1.0, params.upper);
    EXPECT_EQ(params.format, plssvm::file_format_type::libsvm);
    EXPECT_FALSE(params.strings_as_labels);
    EXPECT_FALSE(params.float_as_real_type);
    EXPECT_EQ(params.input_filename, "data.libsvm");
    EXPECT_EQ(params.scaled_filename, "data.libsvm.scaled");
    EXPECT_EQ(params.save_filename, "");
    EXPECT_EQ(params.restore_filename, "");
}
TEST_F(ParameterScale, minimal_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale data.libsvm"));

    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };

    // test output string
    EXPECT_EQ(util::convert_to_string(params), "lower: -1\nupper: 1\nlabel_type: int (default)\nreal_type: double (default)\noutput file format: libsvm\ninput file: 'data.libsvm'\nscaled file: 'data.libsvm.scaled'\nsave file (scaling factors): ''\nrestore file (scaling factors): ''\n");
}

TEST_F(ParameterScale, all_arguments) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale -l -2.0 -u 2.5 -f arff -s data.libsvm.save --use_strings_as_labels --use_float_as_real_type data.libsvm data.libsvm.scaled"));

    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };

    // check default values
    EXPECT_DOUBLE_EQ(-2.0, params.lower);
    EXPECT_DOUBLE_EQ(+2.5, params.upper);
    EXPECT_EQ(params.format, plssvm::file_format_type::arff);
    EXPECT_TRUE(params.strings_as_labels);
    EXPECT_TRUE(params.float_as_real_type);
    EXPECT_EQ(params.input_filename, "data.libsvm");
    EXPECT_EQ(params.scaled_filename, "data.libsvm.scaled");
    EXPECT_EQ(params.save_filename, "data.libsvm.save");
    EXPECT_EQ(params.restore_filename, "");
}
TEST_F(ParameterScale, all_arguments_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale -l -2.0 -u 2.5 -f arff -s data.libsvm.save --use_strings_as_labels --use_float_as_real_type data.libsvm data.libsvm.scaled"));

    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };

    // test output string
    EXPECT_EQ(util::convert_to_string(params), "lower: -2\nupper: 2.5\nlabel_type: std::string\nreal_type: float\noutput file format: arff\ninput file: 'data.libsvm'\nscaled file: 'data.libsvm.scaled'\nsave file (scaling factors): 'data.libsvm.save'\nrestore file (scaling factors): ''\n");
}

// test all command line parameter separately
class ParameterScaleLower : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParameterScaleLower, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.lower, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleLower, ::testing::Combine(
                ::testing::Values("-l", "--lower"),
                ::testing::Values(-2.5, -1.0, -0.01, 0.0)));
// clang-format on

class ParameterScaleUpper : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParameterScaleUpper, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.upper, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleUpper,
                ::testing::Combine(::testing::Values("-u", "--upper"),
                ::testing::Values(0.0, 0.01, 1.0, 2.5)));
// clang-format on

class ParameterScaleFileFormat : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterScaleFileFormat, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to backend
    const auto backend = util::convert_from_string<plssvm::file_format_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.format, backend);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleFileFormat, ::testing::Combine(
                ::testing::Values("-f", "--format"),
                ::testing::Values("libsvm", "LIBSVM", "arff", "ARFF")));
// clang-format on

class ParameterScaleSaveFilename : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterScaleSaveFilename, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.save_filename, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleSaveFilename, ::testing::Combine(
                ::testing::Values("-s", "--save_filename"),
                ::testing::Values("data.libsvm.scaled", "output.txt")));
// clang-format on

class ParameterScaleRestoreFilename : public ParameterScale, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterScaleRestoreFilename, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} {} data.libsvm", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.restore_filename, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleRestoreFilename, ::testing::Combine(
                ::testing::Values("-r", "--restore_filename"),
                ::testing::Values("data.libsvm.weights", "output.txt")));
// clang-format on

class ParameterScaleUseStringsAsLabels : public ParameterScale, public ::testing::WithParamInterface<bool> {};
TEST_P(ParameterScaleUseStringsAsLabels, parsing) {
    const bool value = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale --use_strings_as_labels={} data.libsvm", value));
    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.strings_as_labels, value);
}
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleUseStringsAsLabels, ::testing::Bool());

class ParameterScaleUseFloatAsRealType : public ParameterScale, public ::testing::WithParamInterface<bool> {};
TEST_P(ParameterScaleUseFloatAsRealType, parsing) {
    const bool value = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale --use_float_as_real_type={} data.libsvm", value));
    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.float_as_real_type, value);
}
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleUseFloatAsRealType, ::testing::Bool());

class ParameterScaleQuiet : public ParameterScale, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterScaleQuiet, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-scale {} data.libsvm", flag));
    // create parameter object
    const plssvm::detail::cmd::parameter_scale params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(plssvm::verbose, flag.empty());
}
INSTANTIATE_TEST_SUITE_P(ParameterScale, ParameterScaleQuiet, ::testing::Values("-q", "--quiet", ""));

TEST_F(ParameterScale, help) {
    this->CreateCMDArgs("./plssvm-scale --help");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_scale{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
TEST_F(ParameterScale, version) {
    this->CreateCMDArgs("./plssvm-scale --version");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_scale{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}

TEST_F(ParameterScaleDeathTest, no_positional_argument) {
    this->CreateCMDArgs("./plssvm-scale");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_scale{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_FAILURE), ::testing::StartsWith("Error missing input file!"));
}
TEST_F(ParameterScaleDeathTest, save_and_restore) {
    this->CreateCMDArgs("./plssvm-scale -s data.libsvm.save -r data.libsvm.restore data.libsvm");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_scale{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_FAILURE), ::testing::StartsWith("Error cannot use -s (--save_filename) and -r (--restore_filename) simultaneously!"));
}

TEST_F(ParameterScaleDeathTest, too_many_positional_arguments) {
    this->CreateCMDArgs("./plssvm-scale p1 p2 p3 p4");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_scale{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_FAILURE), ::testing::HasSubstr("Only up to two positional options may be given, but 2 (\"p3 p4\") additional option(s) where provided!"));
}

// test whether nonsensical cmd arguments trigger the assertions
TEST_F(ParameterScaleDeathTest, too_few_argc) {
    EXPECT_DEATH((plssvm::detail::cmd::parameter_scale{ 0, nullptr }), ::testing::HasSubstr("At least one argument is always given (the executable name), but argc is 0!"));
}
TEST_F(ParameterScaleDeathTest, nullptr_argv) {
    EXPECT_DEATH((plssvm::detail::cmd::parameter_scale{ 1, nullptr }), ::testing::HasSubstr("At least one argument is always given (the executable name), but argv is a nullptr!"));
}
TEST_F(ParameterScaleDeathTest, illegal_scaling_range) {
    // illegal [lower, upper] bound range
    this->CreateCMDArgs("./plssvm-scale -l 1.0 -u -1.0 data.libsvm");
    EXPECT_DEATH((plssvm::detail::cmd::parameter_scale{ this->argc, this->argv }), ::testing::HasSubstr("Error invalid scaling range [lower, upper] with [1, -1]!"));
}
TEST_F(ParameterScaleDeathTest, unrecognized_option) {
    this->CreateCMDArgs("./plssvm-scale --foo bar");
    EXPECT_DEATH((plssvm::detail::cmd::parameter_scale{ this->argc, this->argv }), "");
}