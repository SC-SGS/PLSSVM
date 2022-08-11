/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the predict cmd parameter parsing.
 */

#include "plssvm/detail/cmd/parameter_predict.hpp"
#include "plssvm/constants.hpp"  // plssvm::verbose

#include "utility.hpp"  // util::{ParameterBase, convert_to_string, convert_from_string}

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST_F, TEST_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EXIT, EXPECT_DEATH, INSTANTIATE_TEST_SUITE_P,
                                   // ::testing::WithParamInterface, ::testing::Combine, ::testing::Values, ::testing::ExitedWithCode

#include <string>  // std::string
#include <tuple>   // std::tuple

class ParameterPredict : public util::ParameterBase {};
class ParameterPredictDeathTest : public util::ParameterBase {};

TEST_F(ParameterPredict, minimal) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict data.libsvm data.libsvm.model"));

    // create parameter object
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(params.backend, plssvm::backend_type::automatic);
    EXPECT_EQ(params.target, plssvm::target_platform::automatic);
    EXPECT_EQ(params.sycl_implementation_type, plssvm::sycl_generic::implementation_type::automatic);
    EXPECT_FALSE(params.strings_as_labels);
    EXPECT_FALSE(params.float_as_real_type);
    EXPECT_EQ(params.input_filename, "data.libsvm");
    EXPECT_EQ(params.model_filename, "data.libsvm.model");
    EXPECT_EQ(params.predict_filename, "data.libsvm.predict");
}

TEST_F(ParameterPredict, minimal_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict data.libsvm data.libsvm.model"));

    // create parameter object
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };
    // test output string
    EXPECT_EQ(util::convert_to_string(params), "label_type: int (default)\nreal_type: double (default)\ninput file (data set): 'data.libsvm'\ninput file (model): 'data.libsvm.model'\noutput file (prediction): 'data.libsvm.predict'\n");
}

TEST_F(ParameterPredict, all_arguments) {
    // create artificial command line arguments in test fixture
    std::string sycl_specific_flag;
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    sycl_specific_flag = "--sycl_implementation_type dpcpp ";
#endif
    this->CreateCMDArgs(fmt::format("./plssvm-predict --backend cuda --target_platform gpu_nvidia --use_strings_as_labels --use_float_as_real_type {}data.libsvm data.libsvm.model data.libsvm.predict", sycl_specific_flag));

    // create parameter object
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(params.backend, plssvm::backend_type::cuda);
    EXPECT_EQ(params.target, plssvm::target_platform::gpu_nvidia);
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_EQ(params.sycl_implementation_type, plssvm::sycl_generic::implementation_type::dpcpp);
#else
    EXPECT_EQ(params.sycl_implementation_type, plssvm::sycl_generic::implementation_type::automatic);
#endif
    EXPECT_TRUE(params.strings_as_labels);
    EXPECT_TRUE(params.float_as_real_type);
    EXPECT_EQ(params.input_filename, "data.libsvm");
    EXPECT_EQ(params.model_filename, "data.libsvm.model");
    EXPECT_EQ(params.predict_filename, "data.libsvm.predict");
}

TEST_F(ParameterPredict, all_arguments_output) {
    // create artificial command line arguments in test fixture
    std::string sycl_specific_flag;
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    sycl_specific_flag = "--sycl_implementation_type dpcpp ";
#endif
    this->CreateCMDArgs(fmt::format("./plssvm-predict --backend cuda --target_platform gpu_nvidia --use_strings_as_labels --use_float_as_real_type {}data.libsvm data.libsvm.model data.libsvm.predict", sycl_specific_flag));

    // create parameter object
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };
    // test output string
    EXPECT_EQ(util::convert_to_string(params), "label_type: std::string\nreal_type: float\ninput file (data set): 'data.libsvm'\ninput file (model): 'data.libsvm.model'\noutput file (prediction): 'data.libsvm.predict'\n");
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
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.backend, backend);
}
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictBackend, ::testing::Combine(::testing::Values("-b", "--backend"), ::testing::Values("automatic", "OpenMP", "CUDA", "HIP", "OpenCL", "SYCL")));

class ParameterPredictTargetPlatform : public ParameterPredict, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterPredictTargetPlatform, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to target_platform
    const auto target_platform = util::convert_from_string<plssvm::target_platform>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict {} {} data.libsvm data.libsvm.model", flag, value));
    // create parameter object
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.target, target_platform);
}
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictTargetPlatform, ::testing::Combine(::testing::Values("-p", "--target_platform"), ::testing::Values("automatic", "cpu", "gpu_nvidia", "gpu_amd", "gpu_intel")));

#if defined(PLSSVM_HAS_SYCL_BACKEND)

class ParameterPredictSYCLImplementation : public ParameterPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterPredictSYCLImplementation, parsing) {
    // convert string to sycl::implementation_type
    const auto sycl_implementation_type = util::convert_from_string<plssvm::sycl_generic::implementation_type>(GetParam());
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict --sycl_implementation_type={} data.libsvm data.libsvm.model", GetParam()));
    // create parameter object
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.sycl_implementation_type, sycl_implementation_type);
}
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictSYCLImplementation, ::testing::Values("automatic", "hipSYCL", "DPCPP", "DPC++"));

#endif  // PLSSVM_HAS_SYCL_BACKEND

class ParameterPredictUseStringsAsLabels : public ParameterPredict, public ::testing::WithParamInterface<bool> {};
TEST_P(ParameterPredictUseStringsAsLabels, parsing) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict --use_strings_as_labels={} data.libsvm data.libsvm.model", GetParam()));
    // create parameter object
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.strings_as_labels, GetParam());
}
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictUseStringsAsLabels, ::testing::Values(true, false));

class ParameterPredictUseFloatAsRealType : public ParameterPredict, public ::testing::WithParamInterface<bool> {};
TEST_P(ParameterPredictUseFloatAsRealType, parsing) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict --use_float_as_real_type={} data.libsvm data.libsvm.model", GetParam()));
    // create parameter object
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.float_as_real_type, GetParam());
}
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictUseFloatAsRealType, ::testing::Values(true, false));

class ParameterPredictQuiet : public ParameterPredict, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterPredictQuiet, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict {} data.libsvm data.libsvm.model", flag));
    // create parameter object
    plssvm::detail::cmd::parameter_predict params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(plssvm::verbose, flag.empty());
}
INSTANTIATE_TEST_SUITE_P(ParameterPredict, ParameterPredictQuiet, ::testing::Values("-q", "--quiet", ""));

// test whether nonsensical cmd arguments trigger the assertions
TEST_F(ParameterPredictDeathTest, too_few_argc) {
    EXPECT_DEATH((plssvm::detail::cmd::parameter_predict{ 0, nullptr }), ::testing::HasSubstr("At least one argument is always given (the executable name), but argc is 0!"));
}
TEST_F(ParameterPredictDeathTest, nullptr_argv) {
    EXPECT_DEATH((plssvm::detail::cmd::parameter_predict{ 1, nullptr }), ::testing::HasSubstr("At least one argument is always given (the executable name), but argv is a nullptr!"));
}

TEST_F(ParameterPredictDeathTest, help) {
    this->CreateCMDArgs("./plssvm-predict --help");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_predict{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
TEST_F(ParameterPredictDeathTest, version) {
    this->CreateCMDArgs("./plssvm-predict --version");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_predict{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
TEST_F(ParameterPredictDeathTest, no_positional_argument) {
    this->CreateCMDArgs("./plssvm-predict");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_predict{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_FAILURE), "Error missing test file!\n");
}
TEST_F(ParameterPredictDeathTest, single_positional_argument) {
    this->CreateCMDArgs("./plssvm-predict data.libsvm");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_predict{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_FAILURE), "Error missing model file!\n");
}

TEST_F(ParameterPredictDeathTest, too_many_positional_arguments) {
    this->CreateCMDArgs("./plssvm-predict p1 p2 p3 p4");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_predict{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_FAILURE), ::testing::HasSubstr("Only up to three positional options may be given, but 1 (\"p4\") additional option(s) where provided!"));
}

TEST_F(ParameterPredictDeathTest, unrecognized_option) {
    this->CreateCMDArgs("./plssvm-predict --foo bar");
    EXPECT_DEATH((plssvm::detail::cmd::parameter_predict{ this->argc, this->argv }), "");
}