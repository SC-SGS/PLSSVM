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
#include "plssvm/detail/logger.hpp"

#include "plssvm/detail/logger.hpp"  // plssvm::verbosity

#include "../../custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING
#include "../../naming.hpp"              // naming::{pretty_print_parameter_flag_and_value, pretty_print_parameter_flag}
#include "../../utility.hpp"             // util::convert_from_string
#include "utility.hpp"                   // util::ParameterBase

#include "fmt/core.h"                    // fmt::format
#include "gmock/gmock-matchers.h"        // ::testing::{StartsWith, HasSubstr}
#include "gtest/gtest.h"                 // TEST_F, TEST_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EXIT, EXPECT_DEATH, INSTANTIATE_TEST_SUITE_P,
                                         // ::testing::WithParamInterface, ::testing::Combine, ::testing::Values, ::testing::Bool, ::testing::ExitedWithCode

#include <cstdlib>                       // EXIT_SUCCESS, EXIT_FAILURE
#include <string>                        // std::string
#include <tuple>                         // std::tuple

class ParserPredict : public util::ParameterBase {};
class ParserPredictDeathTest : public util::ParameterBase {};

TEST_F(ParserPredict, minimal) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", "data.libsvm", "data.libsvm.model" });

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(parser.backend, plssvm::backend_type::automatic);
    EXPECT_EQ(parser.target, plssvm::target_platform::automatic);
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl::implementation_type::automatic);
    EXPECT_FALSE(parser.strings_as_labels);
    EXPECT_FALSE(parser.float_as_real_type);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.model_filename, "data.libsvm.model");
    EXPECT_EQ(parser.predict_filename, "data.libsvm.predict");
}
TEST_F(ParserPredict, minimal_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", "data.libsvm", "data.libsvm.model" });

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };

    // test output string
    const std::string correct =
        "label_type: int (default)\n"
        "real_type: double (default)\n"
        "input file (data set): 'data.libsvm'\n"
        "input file (model): 'data.libsvm.model'\n"
        "output file (prediction): 'data.libsvm.predict'\n";
    EXPECT_CONVERSION_TO_STRING(parser, correct);
}

TEST_F(ParserPredict, all_arguments) {
    // create artificial command line arguments in test fixture
    std::vector<std::string> cmd_args = { "./plssvm-predict", "--backend", "cuda", "--target_platform", "gpu_nvidia", "--use_strings_as_labels", "--use_float_as_real_type" };
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    cmd_args.insert(cmd_args.end(), { "--sycl_implementation_type", "dpcpp" });
#endif
    cmd_args.insert(cmd_args.end(), { "data.libsvm", "data.libsvm.model", "data.libsvm.predict" });
    this->CreateCMDArgs(cmd_args);

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(parser.backend, plssvm::backend_type::cuda);
    EXPECT_EQ(parser.target, plssvm::target_platform::gpu_nvidia);
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl::implementation_type::dpcpp);
#else
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl::implementation_type::automatic);
#endif
    EXPECT_TRUE(parser.strings_as_labels);
    EXPECT_TRUE(parser.float_as_real_type);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.model_filename, "data.libsvm.model");
    EXPECT_EQ(parser.predict_filename, "data.libsvm.predict");
}
TEST_F(ParserPredict, all_arguments_output) {
    // create artificial command line arguments in test fixture
    std::vector<std::string> cmd_args = { "./plssvm-predict", "--backend", "cuda", "--target_platform", "gpu_nvidia", "--use_strings_as_labels", "--use_float_as_real_type" };
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    cmd_args.insert(cmd_args.end(), { "--sycl_implementation_type", "dpcpp" });
#endif
    cmd_args.insert(cmd_args.end(), { "data.libsvm", "data.libsvm.model", "data.libsvm.predict" });
    this->CreateCMDArgs(cmd_args);

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };

    // test output string
    const std::string correct =
        "label_type: std::string\n"
        "real_type: float\n"
        "input file (data set): 'data.libsvm'\n"
        "input file (model): 'data.libsvm.model'\n"
        "output file (prediction): 'data.libsvm.predict'\n";
    EXPECT_CONVERSION_TO_STRING(parser, correct);
}

// test all command line parameter separately
class ParserPredictBackend : public ParserPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserPredictBackend, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to backend
    const auto backend = util::convert_from_string<plssvm::backend_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag, value, "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.backend, backend);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictBackend, ::testing::Combine(
                ::testing::Values("-b", "--backend"),
                ::testing::Values("automatic", "OpenMP", "CUDA", "HIP", "OpenCL", "SYCL")),
                naming::pretty_print_parameter_flag_and_value<ParserPredictBackend>);
// clang-format on

class ParserPredictTargetPlatform : public ParserPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserPredictTargetPlatform, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to target_platform
    const auto target_platform = util::convert_from_string<plssvm::target_platform>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag, value, "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.target, target_platform);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictTargetPlatform, ::testing::Combine(
                ::testing::Values("-p", "--target_platform"),
                ::testing::Values("automatic", "cpu", "gpu_nvidia", "gpu_amd", "gpu_intel")),
                naming::pretty_print_parameter_flag_and_value<ParserPredictTargetPlatform>);
// clang-format on

#if defined(PLSSVM_HAS_SYCL_BACKEND)

class ParserPredictSYCLImplementation : public ParserPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserPredictSYCLImplementation, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to sycl::implementation_type
    const auto sycl_implementation_type = util::convert_from_string<plssvm::sycl::implementation_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag, value, "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.sycl_implementation_type, sycl_implementation_type);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictSYCLImplementation, ::testing::Combine(
                ::testing::Values("--sycl_implementation_type"),
                ::testing::Values("automatic", "hipSYCL", "DPCPP")),
                naming::pretty_print_parameter_flag_and_value<ParserPredictSYCLImplementation>);
// clang-format on

#endif  // PLSSVM_HAS_SYCL_BACKEND

class ParserPredictUseStringsAsLabels : public ParserPredict, public ::testing::WithParamInterface<std::tuple<std::string, bool>> {};
TEST_P(ParserPredictUseStringsAsLabels, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", fmt::format("{}={}", flag, value), "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.strings_as_labels, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictUseStringsAsLabels, ::testing::Combine(
                ::testing::Values("--use_strings_as_labels"),
                ::testing::Bool()),
                naming::pretty_print_parameter_flag_and_value<ParserPredictUseStringsAsLabels>);
// clang-format on

class ParserPredictUseFloatAsRealType : public ParserPredict, public ::testing::WithParamInterface<std::tuple<std::string, bool>> {};
TEST_P(ParserPredictUseFloatAsRealType, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", fmt::format("{}={}", flag, value), "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.float_as_real_type, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictUseFloatAsRealType, ::testing::Combine(
                ::testing::Values("--use_float_as_real_type"),
                ::testing::Bool()),
                naming::pretty_print_parameter_flag_and_value<ParserPredictUseFloatAsRealType>);
// clang-format on

class ParserPredictVerbosity : public ParserPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserPredictVerbosity, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, value, "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(fmt::format("{}", plssvm::verbosity), value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserPredictVerbosity, ::testing::Combine(
                ::testing::Values("--verbosity"),
                ::testing::Values("quiet", "libsvm", "timing", "full")),
                naming::pretty_print_parameter_flag_and_value<ParserPredictVerbosity>);
// clang-format on

class ParserPredictQuiet : public ParserPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParserPredictQuiet, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag, "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(plssvm::verbosity, flag.empty() ? plssvm::verbosity_level::full : plssvm::verbosity_level::quiet);
}
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictQuiet, ::testing::Values("-q", "--quiet", ""), naming::pretty_print_parameter_flag<ParserPredictQuiet>);

class ParserPredictVerbosityAndQuiet : public ParserPredict {};
TEST_F(ParserPredictVerbosityAndQuiet, parsing) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", "--quiet", "--verbosity", "full", "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->argc, this->argv };
    // the quiet flag overrides the verbosity flag
    EXPECT_EQ(plssvm::verbosity, plssvm::verbosity_level::quiet);
}

class ParserPredictHelp : public ParserPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParserPredictHelp, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag });
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictHelp, ::testing::Values("-h", "--help"), naming::pretty_print_parameter_flag<ParserPredictHelp>);

class ParserPredictVersion : public ParserPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParserPredictVersion, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag });
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictVersion, ::testing::Values("-v", "--version"), naming::pretty_print_parameter_flag<ParserPredictHelp>);

TEST_F(ParserPredict, no_positional_argument) {
    this->CreateCMDArgs({ "./plssvm-predict" });
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::StartsWith("Error missing test file!"));
}
TEST_F(ParserPredict, single_positional_argument) {
    this->CreateCMDArgs({ "./plssvm-predict", "data.libsvm" });
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::StartsWith("Error missing model file!"));
}
TEST_F(ParserPredict, too_many_positional_arguments) {
    this->CreateCMDArgs({ "./plssvm-predict", "p1", "p2", "p3", "p4" });
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::HasSubstr(R"(Only up to three positional options may be given, but 1 ("p4") additional option(s) where provided!)"));
}

// test whether nonsensical cmd arguments trigger the assertions
TEST_F(ParserPredictDeathTest, too_few_argc) {
    EXPECT_DEATH((plssvm::detail::cmd::parser_predict{ 0, nullptr }),
                 ::testing::HasSubstr("At least one argument is always given (the executable name), but argc is 0!"));
}
TEST_F(ParserPredictDeathTest, nullptr_argv) {
    EXPECT_DEATH((plssvm::detail::cmd::parser_predict{ 1, nullptr }),
                 ::testing::HasSubstr("At least one argument is always given (the executable name), but argv is a nullptr!"));
}
TEST_F(ParserPredictDeathTest, unrecognized_option) {
    this->CreateCMDArgs({ "./plssvm-predict", "--foo", "bar" });
    EXPECT_DEATH((plssvm::detail::cmd::parser_predict{ this->argc, this->argv }), "");
}
