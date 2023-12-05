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

#include "plssvm/constants.hpp"        // plssvm::real_type
#include "plssvm/verbosity_levels.hpp"  // plssvm::verbosity

#include "custom_test_macros.hpp"      // EXPECT_CONVERSION_TO_STRING
#include "detail/cmd/cmd_utility.hpp"  // util::ParameterBase
#include "naming.hpp"                  // naming::{pretty_print_parameter_flag_and_value, pretty_print_parameter_flag}
#include "utility.hpp"                 // util::{convert_from_string, redirect_output}

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::{StartsWith, HasSubstr}
#include "gtest/gtest.h"           // TEST_F, TEST_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EXIT, EXPECT_DEATH, INSTANTIATE_TEST_SUITE_P,
                                   // ::testing::WithParamInterface, ::testing::Combine, ::testing::Values, ::testing::Bool, ::testing::ExitedWithCode

#include <cstdlib>      // EXIT_SUCCESS, EXIT_FAILURE
#include <iostream>     // std::clog
#include <string>       // std::string
#include <tuple>        // std::tuple
#include <type_traits>  // std::is_same_v

class ParserPredict : public util::ParameterBase {};

TEST_F(ParserPredict, minimal) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", "data.libsvm", "data.libsvm.model" });

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };

    // check parsed values
    EXPECT_EQ(parser.backend, plssvm::backend_type::automatic);
    EXPECT_EQ(parser.target, plssvm::target_platform::automatic);
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl::implementation_type::automatic);
    EXPECT_FALSE(parser.strings_as_labels);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.model_filename, "data.libsvm.model");
    EXPECT_EQ(parser.predict_filename, "data.libsvm.predict");
    EXPECT_EQ(parser.performance_tracking_filename, "");

    EXPECT_EQ(plssvm::verbosity, plssvm::verbosity_level::full);
}
TEST_F(ParserPredict, minimal_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", "data.libsvm", "data.libsvm.model" });

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };

    // test output string
    const std::string correct = fmt::format(
        "backend: automatic\n"
        "target platform: automatic\n"
        "SYCL implementation type: automatic\n"
        "label_type: int (default)\n"
        "real_type: {}\n"
        "input file (data set): 'data.libsvm'\n"
        "input file (model): 'data.libsvm.model'\n"
        "output file (prediction): 'data.libsvm.predict'\n",
        std::is_same_v<plssvm::real_type, float> ? "float" : "double (default)");

    EXPECT_CONVERSION_TO_STRING(parser, correct);

    EXPECT_EQ(plssvm::verbosity, plssvm::verbosity_level::full);
}

TEST_F(ParserPredict, all_arguments) {
    // create artificial command line arguments in test fixture
    std::vector<std::string> cmd_args = { "./plssvm-predict", "--backend", "cuda", "--target_platform", "gpu_nvidia", "--use_strings_as_labels", "--verbosity", "libsvm" };
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    cmd_args.insert(cmd_args.end(), { "--sycl_implementation_type", "dpcpp" });
#endif
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    cmd_args.insert(cmd_args.end(), { "--performance_tracking", "tracking.yaml" });
#endif
    cmd_args.insert(cmd_args.end(), { "data.libsvm", "data.libsvm.model", "data.libsvm.predict" });
    this->CreateCMDArgs(cmd_args);

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };

    // check parsed values
    EXPECT_EQ(parser.backend, plssvm::backend_type::cuda);
    EXPECT_EQ(parser.target, plssvm::target_platform::gpu_nvidia);
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl::implementation_type::dpcpp);
#else
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl::implementation_type::automatic);
#endif
    EXPECT_TRUE(parser.strings_as_labels);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.model_filename, "data.libsvm.model");
    EXPECT_EQ(parser.predict_filename, "data.libsvm.predict");
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    EXPECT_EQ(parser.performance_tracking_filename, "tracking.yaml");
#else
    EXPECT_EQ(parser.performance_tracking_filename, "");
#endif

    EXPECT_EQ(plssvm::verbosity, plssvm::verbosity_level::libsvm);
}
TEST_F(ParserPredict, all_arguments_output) {
    // create artificial command line arguments in test fixture
    std::vector<std::string> cmd_args = { "./plssvm-predict", "--backend", "cuda", "--target_platform", "gpu_nvidia", "--use_strings_as_labels", "--verbosity", "libsvm" };
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    cmd_args.insert(cmd_args.end(), { "--sycl_implementation_type", "dpcpp" });
#endif
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    cmd_args.insert(cmd_args.end(), { "--performance_tracking", "tracking.yaml" });
#endif
    cmd_args.insert(cmd_args.end(), { "data1.libsvm", "data2.libsvm.model", "data3.libsvm.predict" });
    this->CreateCMDArgs(cmd_args);

    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };

    // test output string
    std::string correct = fmt::format(
        "backend: cuda\n"
        "target platform: gpu_nvidia\n"
        "label_type: std::string\n"
        "real_type: {}\n"
        "input file (data set): 'data1.libsvm'\n"
        "input file (model): 'data2.libsvm.model'\n"
        "output file (prediction): 'data3.libsvm.predict'\n",
        std::is_same_v<plssvm::real_type, float> ? "float" : "double (default)");
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    correct += "performance tracking file: 'tracking.yaml'\n";
#endif

    EXPECT_CONVERSION_TO_STRING(parser, correct);

    EXPECT_EQ(plssvm::verbosity, plssvm::verbosity_level::libsvm);
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
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };
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
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };
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
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };
    // test for correctness
    EXPECT_EQ(parser.sycl_implementation_type, sycl_implementation_type);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictSYCLImplementation, ::testing::Combine(
                ::testing::Values("--sycl_implementation_type"),
                ::testing::Values("automatic", "AdaptiveCpp", "DPCPP")),
                naming::pretty_print_parameter_flag_and_value<ParserPredictSYCLImplementation>);
// clang-format on

#endif  // PLSSVM_HAS_SYCL_BACKEND

#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)

class ParserPredictPerformanceTrackingFilename : public ParserPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserPredictPerformanceTrackingFilename, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag, value, "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };
    // test for correctness
    EXPECT_EQ(parser.performance_tracking_filename, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictPerformanceTrackingFilename, ::testing::Combine(
                ::testing::Values("--performance_tracking"),
                ::testing::Values("tracking.yaml", "test.txt")),
                naming::pretty_print_parameter_flag_and_value<ParserPredictPerformanceTrackingFilename>);
// clang-format on

#endif  // PLSSVM_PERFORMANCE_TRACKER_ENABLED

class ParserPredictUseStringsAsLabels : public ParserPredict, public ::testing::WithParamInterface<std::tuple<std::string, bool>> {};
TEST_P(ParserPredictUseStringsAsLabels, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", fmt::format("{}={}", flag, value), "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };
    // test for correctness
    EXPECT_EQ(parser.strings_as_labels, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictUseStringsAsLabels, ::testing::Combine(
                ::testing::Values("--use_strings_as_labels"),
                ::testing::Bool()),
                naming::pretty_print_parameter_flag_and_value<ParserPredictUseStringsAsLabels>);
// clang-format on

class ParserPredictVerbosity : public ParserPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserPredictVerbosity, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, value, "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };
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
    const plssvm::verbosity_level old_verbosity = plssvm::verbosity;
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag, "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };
    // test for correctness
    EXPECT_EQ(plssvm::verbosity, flag.empty() ? old_verbosity : plssvm::verbosity_level::quiet);
}
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictQuiet, ::testing::Values("-q", "--quiet", ""), naming::pretty_print_parameter_flag<ParserPredictQuiet>);

class ParserPredictVerbosityAndQuiet : public ParserPredict, private util::redirect_output<&std::clog> {};
TEST_F(ParserPredictVerbosityAndQuiet, parsing) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", "--quiet", "--verbosity", "full", "data.libsvm", "data.libsvm.model" });
    // create parameter object
    const plssvm::detail::cmd::parser_predict parser{ this->get_argc(), this->get_argv() };
    // the quiet flag overrides the verbosity flag
    EXPECT_EQ(plssvm::verbosity, plssvm::verbosity_level::quiet);
}

class ParserPredictHelp : public ParserPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParserPredictHelp, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag });
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->get_argc(), this->get_argv() }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictHelp, ::testing::Values("-h", "--help"), naming::pretty_print_parameter_flag<ParserPredictHelp>);

class ParserPredictVersion : public ParserPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParserPredictVersion, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-predict", flag });
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->get_argc(), this->get_argv() }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParserPredict, ParserPredictVersion, ::testing::Values("-v", "--version"), naming::pretty_print_parameter_flag<ParserPredictHelp>);

class ParserPredictDeathTest : public ParserPredict {};

TEST_F(ParserPredictDeathTest, no_positional_argument) {
    this->CreateCMDArgs({ "./plssvm-predict" });
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->get_argc(), this->get_argv() }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::HasSubstr("ERROR: missing test file!"));
}
TEST_F(ParserPredictDeathTest, single_positional_argument) {
    this->CreateCMDArgs({ "./plssvm-predict", "data.libsvm" });
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->get_argc(), this->get_argv() }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::HasSubstr("ERROR: missing model file!"));
}
TEST_F(ParserPredictDeathTest, too_many_positional_arguments) {
    this->CreateCMDArgs({ "./plssvm-predict", "p1", "p2", "p3", "p4" });
    EXPECT_EXIT((plssvm::detail::cmd::parser_predict{ this->get_argc(), this->get_argv() }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::HasSubstr(R"(ERROR: only up to three positional options may be given, but 1 ("p4") additional option(s) where provided!)"));
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
    EXPECT_DEATH((plssvm::detail::cmd::parser_predict{ this->get_argc(), this->get_argv() }), "");
}
