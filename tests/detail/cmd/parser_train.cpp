/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the train cmd parameter parsing.
 */

#include "plssvm/detail/cmd/parser_train.hpp"
#include "plssvm/detail/logger.hpp"

#include "plssvm/detail/logger.hpp"      // plssvm::verbosity

#include "../../custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING
#include "../../naming.hpp"              // naming::{pretty_print_parameter_flag_and_value, pretty_print_parameter_flag}
#include "../../utility.hpp"             // util::convert_from_string
#include "utility.hpp"                   // util::ParameterBase

#include "fmt/core.h"                    // fmt::format
#include "gmock/gmock-matchers.h"        // ::testing::{StartsWith, HasSubstr}
#include "gtest/gtest.h"                 // TEST_F, TEST_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EXIT, EXPECT_DEATH, INSTANTIATE_TEST_SUITE_P,
                                         // ::testing::WithParamInterface, ::testing::Combine, ::testing::Values, ::testing::Range, ::testing::Bool, ::testing::ExitedWithCode

#include <cstddef>                       // std::size_t
#include <cstdlib>                       // EXIT_SUCCESS, EXIT_FAILURE
#include <string>                        // std::string
#include <tuple>                         // std::tuple

class ParserTrain : public util::ParameterBase {};
class ParserTrainDeathTest : public util::ParameterBase {};

TEST_F(ParserTrain, minimal) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", "data.libsvm" });

    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(parser.csvm_params, plssvm::parameter{});
    EXPECT_TRUE(parser.epsilon.is_default());
    EXPECT_DOUBLE_EQ(parser.epsilon.value(), 0.001);
    EXPECT_TRUE(parser.max_iter.is_default());
    EXPECT_EQ(parser.max_iter.value(), 0);
    EXPECT_EQ(parser.backend, plssvm::backend_type::automatic);
    EXPECT_EQ(parser.target, plssvm::target_platform::automatic);
    EXPECT_EQ(parser.sycl_kernel_invocation_type, plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl::implementation_type::automatic);
    EXPECT_FALSE(parser.strings_as_labels);
    EXPECT_FALSE(parser.float_as_real_type);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.model_filename, "data.libsvm.model");
    EXPECT_EQ(parser.performance_tracking_filename, "");
}
TEST_F(ParserTrain, minimal_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", "data.libsvm" });

    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };

    // test output string
    const std::string correct =
        "kernel_type: linear -> u'*v\n"
        "cost: 1 (default)\n"
        "epsilon: 0.001 (default)\n"
        "max_iter: num_data_points (default)\n"
        "label_type: int (default)\n"
        "real_type: double (default)\n"
        "input file (data set): 'data.libsvm'\n"
        "output file (model): 'data.libsvm.model'\n"
        "performance tracking file: ''\n";
    EXPECT_CONVERSION_TO_STRING(parser, correct);
}

TEST_F(ParserTrain, all_arguments) {
    // create artificial command line arguments in test fixture
    std::vector<std::string> cmd_args = { "./plssvm-train", "--kernel_type", "1", "--degree", "2", "--gamma", "1.5", "--coef0", "-1.5", "--cost", "2", "--epsilon", "1e-10", "--max_iter", "100", "--backend", "cuda", "--target_platform", "gpu_nvidia", "--use_strings_as_labels", "--use_float_as_real_type", "--verbosity", "libsvm" };
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    cmd_args.insert(cmd_args.end(), { "--sycl_kernel_invocation_type", "nd_range", "--sycl_implementation_type", "dpcpp" });
#endif
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    cmd_args.insert(cmd_args.end(), { "--performance_tracking", "tracking.yaml" });
#endif
    cmd_args.insert(cmd_args.end(), { "data.libsvm", "data.libsvm.model" });
    this->CreateCMDArgs(cmd_args);

    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(parser.csvm_params.kernel_type, plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(parser.csvm_params.degree, 2);
    EXPECT_DOUBLE_EQ(parser.csvm_params.gamma, 1.5);
    EXPECT_DOUBLE_EQ(parser.csvm_params.coef0, -1.5);
    EXPECT_DOUBLE_EQ(parser.csvm_params.cost, 2.0);

    EXPECT_FALSE(parser.epsilon.is_default());
    EXPECT_DOUBLE_EQ(parser.epsilon.value(), 1e-10);
    EXPECT_FALSE(parser.max_iter.is_default());
    EXPECT_EQ(parser.max_iter.value(), 100);
    EXPECT_EQ(parser.backend, plssvm::backend_type::cuda);
    EXPECT_EQ(parser.target, plssvm::target_platform::gpu_nvidia);
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_EQ(parser.sycl_kernel_invocation_type, plssvm::sycl::kernel_invocation_type::nd_range);
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl::implementation_type::dpcpp);
#else
    EXPECT_EQ(parser.sycl_kernel_invocation_type, plssvm::sycl::kernel_invocation_type::automatic);
    EXPECT_EQ(parser.sycl_implementation_type, plssvm::sycl::implementation_type::automatic);
#endif
    EXPECT_TRUE(parser.strings_as_labels);
    EXPECT_TRUE(parser.float_as_real_type);
    EXPECT_EQ(parser.input_filename, "data.libsvm");
    EXPECT_EQ(parser.model_filename, "data.libsvm.model");
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    EXPECT_EQ(parser.performance_tracking_filename, "tracking.yaml");
#else
    EXPECT_EQ(parser.performance_tracking_filename, "");
#endif
    EXPECT_EQ(plssvm::verbosity, plssvm::verbosity_level::libsvm);
}
TEST_F(ParserTrain, all_arguments_output) {
    // create artificial command line arguments in test fixture
    std::vector<std::string> cmd_args = { "./plssvm-train", "--kernel_type", "1", "--degree", "2", "--gamma", "1.5", "--coef0", "-1.5", "--cost", "2", "--epsilon", "1e-10", "--max_iter", "100", "--backend", "cuda", "--target_platform", "gpu_nvidia", "--use_strings_as_labels", "--use_float_as_real_type", "--verbosity", "libsvm" };
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    cmd_args.insert(cmd_args.end(), { "--sycl_kernel_invocation_type", "nd_range", "--sycl_implementation_type", "dpcpp" });
#endif
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
    cmd_args.insert(cmd_args.end(), { "--performance_tracking", "tracking.yaml" });
#endif
    cmd_args.insert(cmd_args.end(), { "data.libsvm", "data.libsvm.model" });
    this->CreateCMDArgs(cmd_args);

    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };

    // test output string
    const std::string correct =
        "kernel_type: polynomial -> (gamma*u'*v+coef0)^degree\n"
        "gamma: 1.5\n"
        "coef0: -1.5\n"
        "degree: 2\n"
        "cost: 2\n"
        "epsilon: 1e-10\n"
        "max_iter: 100\n"
        "label_type: std::string\n"
        "real_type: float\n"
        "input file (data set): 'data.libsvm'\n"
        "output file (model): 'data.libsvm.model'\n"
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)
        "performance tracking file: 'tracking.yaml'\n";
#else
        "performance tracking file: ''\n";
#endif
    EXPECT_CONVERSION_TO_STRING(parser, correct);
    EXPECT_EQ(plssvm::verbosity, plssvm::verbosity_level::libsvm);
}

// test all command line parameter separately
class ParserTrainKernel : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserTrainKernel, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to kernel_type
    const auto kernel_type = util::convert_from_string<plssvm::kernel_function_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, value, "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(parser.csvm_params.kernel_type.is_default());
    EXPECT_EQ(parser.csvm_params.kernel_type, kernel_type);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainKernel, ::testing::Combine(
                ::testing::Values("-t", "--kernel_type"),
                ::testing::Values("linear", "0", "polynomial", "1", "rbf", "2")),
                naming::pretty_print_parameter_flag_and_value<ParserTrainKernel>);
// clang-format on

class ParserTrainDegree : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, int>> {};
TEST_P(ParserTrainDegree, parsing) {
    const auto &[flag, degree] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, fmt::format("{}", degree), "data.libsvm" });
    ;
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(parser.csvm_params.degree.is_default());
    EXPECT_EQ(parser.csvm_params.degree, degree);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainDegree, ::testing::Combine(
                ::testing::Values("-d", "--degree"),
                ::testing::Range(-2, 3)),
                naming::pretty_print_parameter_flag_and_value<ParserTrainDegree>);
// clang-format on

class ParserTrainGamma : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParserTrainGamma, parsing) {
    const auto &[flag, gamma] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, fmt::format("{}", gamma), "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(parser.csvm_params.gamma.is_default());
    EXPECT_DOUBLE_EQ(parser.csvm_params.gamma, gamma);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainGamma,
                ::testing::Combine(::testing::Values("-g", "--gamma"),
                ::testing::Values(0.001, 0.75, 1.5, 2)),
                naming::pretty_print_parameter_flag_and_value<ParserTrainGamma>);
// clang-format on

class ParserTrainGammaDeathTest : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParserTrainGammaDeathTest, gamma_explicit_less_or_equal_to_zero) {
    const auto &[flag, gamma] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, fmt::format("{}", gamma), "data.libsvm" });
    // create parser_train object
    EXPECT_DEATH((plssvm::detail::cmd::parser_train{ this->argc, this->argv }), ::testing::HasSubstr(fmt::format("gamma must be greater than 0.0, but is {}!", gamma)));
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrainDeathTest, ParserTrainGammaDeathTest, ::testing::Combine(
                ::testing::Values("-g", "--gamma"),
                ::testing::Values(-2, -1.5, 0.0)),
                naming::pretty_print_parameter_flag_and_value<ParserTrainGammaDeathTest>);
// clang-format on

class ParserTrainCoef0 : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParserTrainCoef0, parsing) {
    const auto &[flag, coef0] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, fmt::format("{}", coef0), "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(parser.csvm_params.coef0.is_default());
    EXPECT_DOUBLE_EQ(parser.csvm_params.coef0, coef0);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainCoef0, ::testing::Combine(
                ::testing::Values("-r", "--coef0"),
                ::testing::Values(-2, -1.5, -0.75, -0.001, 0, 0.001, 0.75, 1.5, 2)),
                naming::pretty_print_parameter_flag_and_value<ParserTrainCoef0>);
// clang-format on

class ParserTrainCost : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, int>> {};
TEST_P(ParserTrainCost, parsing) {
    const auto &[flag, cost] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, fmt::format("{}", cost), "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(parser.csvm_params.cost.is_default());
    EXPECT_DOUBLE_EQ(parser.csvm_params.cost, cost);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainCost, ::testing::Combine(
                ::testing::Values("-c", "--cost"),
                ::testing::Range(-2, 3)),
                naming::pretty_print_parameter_flag_and_value<ParserTrainCost>);
// clang-format on

class ParserTrainEpsilon : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParserTrainEpsilon, parsing) {
    const auto &[flag, eps] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, fmt::format("{}", eps), "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(parser.epsilon.is_default());
    EXPECT_DOUBLE_EQ(parser.epsilon, eps);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainEpsilon, ::testing::Combine(
                ::testing::Values("-e", "--epsilon"),
                ::testing::Values(10.0, 1.0, 0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001)),
                naming::pretty_print_parameter_flag_and_value<ParserTrainEpsilon>);
// clang-format on

class ParserTrainMaxIter : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::size_t>> {};
TEST_P(ParserTrainMaxIter, parsing) {
    const auto &[flag, max_iter] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, fmt::format("{}", max_iter), "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(parser.max_iter.is_default());
    EXPECT_EQ(parser.max_iter, max_iter);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainMaxIter, ::testing::Combine(
                ::testing::Values("-i", "--max_iter"),
                ::testing::Values(1, 10, 100, 1000, 10000)),
                naming::pretty_print_parameter_flag_and_value<ParserTrainMaxIter>);
// clang-format on

class ParserTrainMaxIterDeathTest : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, long long int>> {};
TEST_P(ParserTrainMaxIterDeathTest, max_iter_explicit_less_or_equal_to_zero) {
    const auto &[flag, max_iter] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, fmt::format("{}", max_iter), "data.libsvm" });
    // create parameter object
    EXPECT_DEATH((plssvm::detail::cmd::parser_train{ this->argc, this->argv }), ::testing::HasSubstr(fmt::format("max_iter must be greater than 0, but is {}!", max_iter)));
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrainDeathTest, ParserTrainMaxIterDeathTest, ::testing::Combine(
                ::testing::Values("-i", "--max_iter"),
                ::testing::Values(-100, -10, -1, 0)),
                naming::pretty_print_parameter_flag_and_value<ParserTrainMaxIterDeathTest>);
// clang-format on

class ParserTrainBackend : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserTrainBackend, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to backend
    const auto backend = util::convert_from_string<plssvm::backend_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, value, "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.backend, backend);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainBackend, ::testing::Combine(
                ::testing::Values("-b", "--backend"),
                ::testing::Values("automatic", "OpenMP", "CUDA", "HIP", "OpenCL", "SYCL")),
                naming::pretty_print_parameter_flag_and_value<ParserTrainBackend>);
// clang-format on

class ParserTrainTargetPlatform : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserTrainTargetPlatform, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to target_platform
    const auto target_platform = util::convert_from_string<plssvm::target_platform>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, value, "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.target, target_platform);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainTargetPlatform, ::testing::Combine(
                ::testing::Values("-p", "--target_platform"),
                ::testing::Values("automatic", "cpu", "gpu_nvidia", "gpu_amd", "gpu_intel")),
                naming::pretty_print_parameter_flag_and_value<ParserTrainTargetPlatform>);
// clang-format on

#if defined(PLSSVM_HAS_SYCL_BACKEND)

class ParserTrainSYCLKernelInvocation : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserTrainSYCLKernelInvocation, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to sycl::kernel_invocation_type
    const auto sycl_kernel_invocation_type = util::convert_from_string<plssvm::sycl::kernel_invocation_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, value, "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.sycl_kernel_invocation_type, sycl_kernel_invocation_type);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainSYCLKernelInvocation, ::testing::Combine(
                ::testing::Values("--sycl_kernel_invocation_type"),
                ::testing::Values("automatic", "nd_range", "ND_RANGE", "hierarchical")),
                naming::pretty_print_parameter_flag_and_value<ParserTrainSYCLKernelInvocation>);
// clang-format on

class ParserTrainSYCLImplementation : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserTrainSYCLImplementation, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to sycl::implementation_type
    const auto sycl_implementation_type = util::convert_from_string<plssvm::sycl::implementation_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, value, "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.sycl_implementation_type, sycl_implementation_type);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainSYCLImplementation, ::testing::Combine(
                ::testing::Values("--sycl_implementation_type"),
                ::testing::Values("automatic", "hipSYCL", "DPCPP")),
                naming::pretty_print_parameter_flag_and_value<ParserTrainSYCLImplementation>);
// clang-format on

#endif  // PLSSVM_HAS_SYCL_BACKEND

#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)

class ParserTrainPerformanceTrackingFilename : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserTrainPerformanceTrackingFilename, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, value, "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.performance_tracking_filename, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainPerformanceTrackingFilename, ::testing::Combine(
                ::testing::Values("--performance_tracking"),
                ::testing::Values("tracking.yaml", "test.txt")),
                naming::pretty_print_parameter_flag_and_value<ParserTrainPerformanceTrackingFilename>);
// clang-format on

#endif  // PLSSVM_PERFORMANCE_TRACKER_ENABLED

class ParserTrainUseStringsAsLabels : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, bool>> {};
TEST_P(ParserTrainUseStringsAsLabels, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", fmt::format("{}={}", flag, value), "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.strings_as_labels, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainUseStringsAsLabels, ::testing::Combine(
                ::testing::Values("--use_strings_as_labels"),
                ::testing::Bool()),
                naming::pretty_print_parameter_flag_and_value<ParserTrainUseStringsAsLabels>);
// clang-format on

class ParserTrainUseFloatAsRealType : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, bool>> {};
TEST_P(ParserTrainUseFloatAsRealType, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", fmt::format("{}={}", flag, value), "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(parser.float_as_real_type, value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainUseFloatAsRealType, ::testing::Combine(
                ::testing::Values("--use_float_as_real_type"),
                ::testing::Bool()),
                naming::pretty_print_parameter_flag_and_value<ParserTrainUseFloatAsRealType>);
// clang-format on

class ParserTrainVerbosity : public ParserTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParserTrainVerbosity, parsing) {
    const auto &[flag, value] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, value, "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(fmt::format("{}", plssvm::verbosity), value);
}
// clang-format off
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainVerbosity, ::testing::Combine(
                ::testing::Values("--verbosity"),
                ::testing::Values("quiet", "libsvm", "timing", "full")),
                naming::pretty_print_parameter_flag_and_value<ParserTrainVerbosity>);
// clang-format on

class ParserTrainQuiet : public ParserTrain, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParserTrainQuiet, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag, "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(plssvm::verbosity, flag.empty() ? plssvm::verbosity_level::full : plssvm::verbosity_level::quiet);
}
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainQuiet, ::testing::Values("-q", "--quiet", ""), naming::pretty_print_parameter_flag<ParserTrainQuiet>);

class ParserTrainVerbosityAndQuiet : public ParserTrain {};
TEST_F(ParserTrainVerbosityAndQuiet, parsing) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", "--quiet", "--verbosity", "full", "data.libsvm" });
    // create parameter object
    const plssvm::detail::cmd::parser_train parser{ this->argc, this->argv };
    // the quiet flag overrides the verbosity flag
    EXPECT_EQ(plssvm::verbosity, plssvm::verbosity_level::quiet);
}

class ParserTrainHelp : public ParserTrain, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParserTrainHelp, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag });
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_train{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainHelp, ::testing::Values("-h", "--help"), naming::pretty_print_parameter_flag<ParserTrainHelp>);

class ParserTrainVersion : public ParserTrain, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParserTrainVersion, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs({ "./plssvm-train", flag });
    // create parameter object
    EXPECT_EXIT((plssvm::detail::cmd::parser_train{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
INSTANTIATE_TEST_SUITE_P(ParserTrain, ParserTrainVersion, ::testing::Values("-v", "--version"), naming::pretty_print_parameter_flag<ParserTrainVersion>);

TEST_F(ParserTrain, no_positional_argument) {
    this->CreateCMDArgs({ "./plssvm-train" });
    EXPECT_EXIT((plssvm::detail::cmd::parser_train{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::StartsWith("Error missing input file!"));
}
TEST_F(ParserTrain, too_many_positional_arguments) {
    this->CreateCMDArgs({ "./plssvm-train", "p1", "p2", "p3", "p4" });
    EXPECT_EXIT((plssvm::detail::cmd::parser_train{ this->argc, this->argv }),
                ::testing::ExitedWithCode(EXIT_FAILURE),
                ::testing::HasSubstr(R"(Only up to two positional options may be given, but 2 ("p3 p4") additional option(s) where provided!)"));
}

// test whether nonsensical cmd arguments trigger the assertions
TEST_F(ParserTrainDeathTest, too_few_argc) {
    EXPECT_DEATH((plssvm::detail::cmd::parser_train{ 0, nullptr }),
                 ::testing::HasSubstr("At least one argument is always given (the executable name), but argc is 0!"));
}
TEST_F(ParserTrainDeathTest, nullptr_argv) {
    EXPECT_DEATH((plssvm::detail::cmd::parser_train{ 1, nullptr }),
                 ::testing::HasSubstr("At least one argument is always given (the executable name), but argv is a nullptr!"));
}
TEST_F(ParserTrainDeathTest, unrecognized_option) {
    this->CreateCMDArgs({ "./plssvm-train", "--foo", "bar" });
    EXPECT_DEATH((plssvm::detail::cmd::parser_train{ this->argc, this->argv }), "");
}