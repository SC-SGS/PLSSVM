/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the predict cmd parameter parsing.
 */

#include "plssvm/detail/cmd/parser_predict.hpp"

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

class ParameterPredict : public util::ParameterBase {};
class ParameterPredictDeathTest : public util::ParameterBase {};

TEST_F(ParameterPredict, minimal) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict data.libsvm data.libsvm.model"));

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(parser.backend, plssvm::backend_type::automatic);
    EXPECT_EQ(parser.target, plssvm::target_platform::automatic);
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl_generic::implementation_type::automatic);
    EXPECT_FALSE(parser.strings_as_labels);
    EXPECT_FALSE(parser.float_as_real_type);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.model_filename, "data.libsvm.model");
    EXPECT_EQ(parser.predict_filename, "data.libsvm.predict");
}
TEST_F(ParameterPredict, minimal_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict data.libsvm data.libsvm.model"));

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };

    // test output string
    const std::string correct =
        "label_type: int (default)\n"
        "real_type: double (default)\n"
        "input file (data set): 'data.libsvm'\n"
        "input file (model): 'data.libsvm.model'\n"
        "output file (prediction): 'data.libsvm.predict'\n";
    EXPECT_EQ(util::convert_to_string(parser), correct);
}

TEST_F(ParameterPredict, all_arguments) {
    // create artificial command line arguments in test fixture
    std::string sycl_specific_flag;
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    sycl_specific_flag = "--sycl_implementation_type dpcpp ";
#endif
    this->CreateCMDArgs(fmt::format("./plssvm-predict --backend cuda --target_platform gpu_nvidia --use_strings_as_labels --use_float_as_real_type {}data.libsvm data.libsvm.model data.libsvm.predict", sycl_specific_flag));

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(parser.backend, plssvm::backend_type::cuda);
    EXPECT_EQ(parser.target, plssvm::target_platform::gpu_nvidia);
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl_generic::implementation_type::dpcpp);
#else
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl_generic::implementation_type::automatic);
#endif
    EXPECT_TRUE(parser.strings_as_labels);
    EXPECT_TRUE(parser.float_as_real_type);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.model_filename, "data.libsvm.model");
    EXPECT_EQ(parser.predict_filename, "data.libsvm.predict");
}
TEST_F(ParameterPredict, all_arguments_output) {
    // create artificial command line arguments in test fixture
    std::string sycl_specific_flag;
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    sycl_specific_flag = "--sycl_implementation_type dpcpp ";
#endif
    this->CreateCMDArgs(fmt::format("./plssvm-predict --backend cuda --target_platform gpu_nvidia --use_strings_as_labels --use_float_as_real_type {}data.libsvm data.libsvm.model data.libsvm.predict", sycl_specific_flag));

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };

    // test output string
    const std::string correct =
        "label_type: std::string\n"
        "real_type: float\n"
        "input file (data set): 'data.libsvm'\n"
        "input file (model): 'data.libsvm.model'\n"
        "output file (prediction): 'data.libsvm.predict'\n";
    EXPECT_EQ(util::convert_to_string(parser), correct);
}

// test all command line parameter separately
class ParameterPredictBackend : public ParameterPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterPredictBackend, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to backend
    const auto backend = util::convert_from_string<plssvm::backend_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict {} {} data.libsvm data.libsvm.model", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.backend, backend);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictBackend, ::testing::Combine(
                ::testing::Values("-b", "--backend"),
                ::testing::Values("automatic", "OpenMP", "CUDA", "HIP", "OpenCL", "SYCL")),
                naming::pretty_print_parameter_flag_and_value<ParameterPredictBackend>);
// clang-format on

class ParameterPredictTargetPlatform : public ParameterPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterPredictTargetPlatform, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to target_platform
    const auto target_platform = util::convert_from_string<plssvm::target_platform>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict {} {} data.libsvm data.libsvm.model", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.target, target_platform);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictTargetPlatform, ::testing::Combine(
                ::testing::Values("-p", "--target_platform"),
                ::testing::Values("automatic", "cpu", "gpu_nvidia", "gpu_amd", "gpu_intel")),
                naming::pretty_print_parameter_flag_and_value<ParameterPredictTargetPlatform>);
// clang-format on

#if defined(PLSSVM_HAS_SYCL_BACKEND)

class ParameterPredictSYCLImplementation : public ParameterPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterPredictSYCLImplementation, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to sycl::implementation_type
    const auto sycl_implementation_type = util::convert_from_string<plssvm::sycl_generic::implementation_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict {}={} data.libsvm data.libsvm.model", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.sycl_implementation_type, sycl_implementation_type);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictSYCLImplementation, ::testing::Combine(
                ::testing::Values("--sycl_implementation_type"),
                ::testing::Values("automatic", "hipSYCL", "DPCPP", "DPC++")),
                naming::pretty_print_parameter_flag_and_value<ParameterPredictSYCLImplementation>);
// clang-format on

#endif  // PLSSVM_HAS_SYCL_BACKEND

class ParameterPredictUseStringsAsLabels : public ParameterPredict, public ::testing::WithParamInterface<std::tuple<std::string, bool>> {};
TEST_P(ParameterPredictUseStringsAsLabels, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    fmt::print("./plssvm-predict {} {} data.libsvm data.libsvm.model", flag, value);
    this->CreateCMDArgs(fmt::format("./plssvm-predict {}={} data.libsvm data.libsvm.model", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.strings_as_labels, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictUseStringsAsLabels, ::testing::Combine(
                ::testing::Values("--use_strings_as_labels"),
                ::testing::Bool()),
                naming::pretty_print_parameter_flag_and_value<ParameterPredictUseStringsAsLabels>);
// clang-format on

class ParameterPredictUseFloatAsRealType : public ParameterPredict, public ::testing::WithParamInterface<std::tuple<std::string, bool>> {};
TEST_P(ParameterPredictUseFloatAsRealType, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict {}={} data.libsvm data.libsvm.model", flag, value));
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.float_as_real_type, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictUseFloatAsRealType, ::testing::Combine(
                ::testing::Values("--use_float_as_real_type"),
                ::testing::Bool()),
                naming::pretty_print_parameter_flag_and_value<ParameterPredictUseFloatAsRealType>);
// clang-format on

class ParameterPredictQuiet : public ParameterPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterPredictQuiet, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict {} data.libsvm data.libsvm.model", flag));
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(plssvm::verbose, flag.empty());
}
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictQuiet, ::testing::Values("-q", "--quiet", ""), naming::pretty_print_parameter_flag<ParameterPredictQuiet>);

class ParameterPredictHelp : public ParameterPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterPredictHelp, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict {}", flag));
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictHelp, ::testing::Values("-h", "--help"), naming::pretty_print_parameter_flag<ParameterPredictHelp>);

class ParameterPredictVersion : public ParameterPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterPredictVersion, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict {}", flag));
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictVersion, ::testing::Values("-v", "--version"), naming::pretty_print_parameter_flag<ParameterPredictHelp>);

TEST_F(ParameterPredict, no_positional_argument) {
    this->CreateCMDArgs("./plssvm-predict");
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::StartsWith("Error missing test file!"));
}
TEST_F(ParameterPredict, single_positional_argument) {
    this->CreateCMDArgs("./plssvm-predict data.libsvm");
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::StartsWith("Error missing model file!"));
}
TEST_F(ParameterPredict, too_many_positional_arguments) {
    this->CreateCMDArgs("./plssvm-predict p1 p2 p3 p4");
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::HasSubstr(R"(Only up to three positional options may be given, but 1 ("p4") additional option(s) where provided!)"));
}

// test whether nonsensical cmd arguments trigger the assertions
TEST_F(ParameterPredictDeathTest, too_few_argc) {
    EXPECT_DEATH((plssvm::detail::cmd::parser_predict{ 0, nullptr }),
                 ::testing::HasSubstr("At least one argument is always given (the executable name), but argc is 0!"));
}
TEST_F(ParameterPredictDeathTest, nullptr_argv) {
    EXPECT_DEATH((plssvm::detail::cmd::parser_predict{ 1, nullptr }),
                 ::testing::HasSubstr("At least one argument is always given (the executable name), but argv is a nullptr!"));
}
TEST_F(ParameterPredictDeathTest, unrecognized_option) {
    this->CreateCMDArgs("./plssvm-predict --foo bar");
    EXPECT_DEATH((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }), "");
}
