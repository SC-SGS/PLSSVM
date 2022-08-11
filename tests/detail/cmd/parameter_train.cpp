/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the train cmd parameter parsing.
 */

#include "plssvm/detail/cmd/parameter_train.hpp"
#include "plssvm/constants.hpp"  // plssvm::verbose

#include "utility.hpp"  // util::{ParameterBase, convert_to_string, convert_from_string}

#include "fmt/core.h"              // fmt::format
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST_F, TEST_P, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_EXIT, EXPECT_DEATH, INSTANTIATE_TEST_SUITE_P,
                                   // ::testing::WithParamInterface, ::testing::Combine, ::testing::Values, ::testing::Range, ::testing::ExitedWithCode

#include <string>  // std::string
#include <tuple>   // std::tuple

class ParameterTrain : public util::ParameterBase {};
class ParameterTrainDeathTest : public util::ParameterBase {};

TEST_F(ParameterTrain, minimal) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train data.libsvm"));

    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(params.csvm_params, plssvm::parameter<double>{});
    EXPECT_TRUE(params.epsilon.is_default());
    EXPECT_EQ(params.epsilon.value(), 0.001);
    EXPECT_TRUE(params.max_iter.is_default());
    EXPECT_EQ(params.max_iter.value(), 0);
    EXPECT_EQ(params.backend, plssvm::backend_type::automatic);
    EXPECT_EQ(params.target, plssvm::target_platform::automatic);
    EXPECT_EQ(params.sycl_kernel_invocation_type, plssvm::sycl_generic::kernel_invocation_type::automatic);
    EXPECT_EQ(params.sycl_implementation_type, plssvm::sycl_generic::implementation_type::automatic);
    EXPECT_FALSE(params.strings_as_labels);
    EXPECT_FALSE(params.float_as_real_type);
    EXPECT_EQ(params.input_filename, "data.libsvm");
    EXPECT_EQ(params.model_filename, "data.libsvm.model");
}
TEST_F(ParameterTrain, minimal_output) {
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train data.libsvm"));

    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };

    // test output string
    std::string correct =
        "kernel_type: linear -> u'*v\n"
        "cost: 1 (default)\n"
        "epsilon: 0.001 (default)\n"
        "max_iter: num_data_points (default)\n"
        "label_type: int (default)\n"
        "real_type: double (default)\n"
        "input file (data set): 'data.libsvm'\n"
        "output file (model): 'data.libsvm.model'\n";
    EXPECT_EQ(util::convert_to_string(params), correct);
}

TEST_F(ParameterTrain, all_arguments) {
    // create artificial command line arguments in test fixture
    std::string sycl_specific_flag;
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    sycl_specific_flag = "--sycl_kernel_invocation_type nd_range --sycl_implementation_type dpcpp ";
#endif
    this->CreateCMDArgs(fmt::format("./plssvm-train --kernel_type 1 --degree 2 --gamma 1.5 --coef0 -1.5 --cost 2 --epsilon 1e-10 --max_iter 100 --backend cuda --target_platform gpu_nvidia --use_strings_as_labels --use_float_as_real_type {}data.libsvm data.libsvm.model", sycl_specific_flag));

    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };

    // check parsed values
    EXPECT_EQ(params.csvm_params.kernel, plssvm::kernel_type::polynomial);
    EXPECT_EQ(params.csvm_params.degree, 2);
    EXPECT_EQ(params.csvm_params.gamma, 1.5);
    EXPECT_EQ(params.csvm_params.coef0, -1.5);
    EXPECT_EQ(params.csvm_params.cost, 2);

    EXPECT_FALSE(params.epsilon.is_default());
    EXPECT_EQ(params.epsilon.value(), 1e-10);
    EXPECT_FALSE(params.max_iter.is_default());
    EXPECT_EQ(params.max_iter.value(), 100);
    EXPECT_EQ(params.backend, plssvm::backend_type::cuda);
    EXPECT_EQ(params.target, plssvm::target_platform::gpu_nvidia);
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_EQ(params.sycl_kernel_invocation_type, plssvm::sycl_generic::kernel_invocation_type::nd_range);
    EXPECT_EQ(params.sycl_implementation_type, plssvm::sycl_generic::implementation_type::dpcpp);
#else
    EXPECT_EQ(params.sycl_kernel_invocation_type, plssvm::sycl_generic::kernel_invocation_type::automatic);
    EXPECT_EQ(params.sycl_implementation_type, plssvm::sycl_generic::implementation_type::automatic);
#endif
    EXPECT_TRUE(params.strings_as_labels);
    EXPECT_TRUE(params.float_as_real_type);
    EXPECT_EQ(params.input_filename, "data.libsvm");
    EXPECT_EQ(params.model_filename, "data.libsvm.model");
}

TEST_F(ParameterTrain, all_arguments_output) {
    // create artificial command line arguments in test fixture
    std::string sycl_specific_flag;
#if defined(PLSSVM_HAS_SYCL_BACKEND)
    sycl_specific_flag = "--sycl_kernel_invocation_type nd_range --sycl_implementation_type dpcpp ";
#endif
    this->CreateCMDArgs(fmt::format("./plssvm-train --kernel_type 1 --degree 2 --gamma 1.5 --coef0 -1.5 --cost 2 --epsilon 1e-10 --max_iter 100 --backend cuda --target_platform gpu_nvidia --use_strings_as_labels --use_float_as_real_type {}data.libsvm data.libsvm.model", sycl_specific_flag));

    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };

    // test output string
    std::string correct =
        "kernel_type: polynomial -> (gamma*u'*v + coef0)^degree\n"
        "gamma: 1.5\n"
        "coef0: -1.5\n"
        "degree: 2\n"
        "cost: 2\n"
        "epsilon: 1e-10\n"
        "max_iter: 100\n"
        "label_type: std::string\n"
        "real_type: float\n"
        "input file (data set): 'data.libsvm'\n"
        "output file (model): 'data.libsvm.model'\n";
    EXPECT_EQ(util::convert_to_string(params), correct);
}

// test all command line parameter separately
class ParameterTrainKernel : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterTrainKernel, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to kernel_type
    const auto kernel_type = util::convert_from_string<plssvm::kernel_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, value));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(params.csvm_params.kernel.is_default());
    EXPECT_EQ(params.csvm_params.kernel, kernel_type);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainKernel, ::testing::Combine(::testing::Values("-t", "--kernel_type"), ::testing::Values("linear", "0", "polynomial", "1", "rbf", "2")));

class ParameterTrainDegree : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, int>> {};
TEST_P(ParameterTrainDegree, parsing) {
    const auto &[flag, degree] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, degree));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(params.csvm_params.degree.is_default());
    EXPECT_EQ(params.csvm_params.degree, degree);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainDegree, ::testing::Combine(::testing::Values("-d", "--degree"), ::testing::Range(-2, 3)));

class ParameterTrainGamma : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParameterTrainGamma, parsing) {
    const auto &[flag, gamma] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, gamma));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(params.csvm_params.gamma.is_default());
    EXPECT_EQ(params.csvm_params.gamma, gamma);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainGamma, ::testing::Combine(::testing::Values("-g", "--gamma"), ::testing::Values(0.001, 0.75, 1.5, 2)));

class ParameterTrainGammaDeathTest : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParameterTrainGammaDeathTest, gamma_explicit_less_or_equal_to_zero) {
    const auto &[flag, gamma] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, gamma));
    // create parameter_train object
    EXPECT_DEATH((plssvm::detail::cmd::parameter_train{ this->argc, this->argv }), ::testing::HasSubstr(fmt::format("gamma must be greater than 0.0, but is {}!", gamma)));
}
INSTANTIATE_TEST_SUITE_P(ParameterTrainDeathTest, ParameterTrainGammaDeathTest, ::testing::Combine(::testing::Values("-g", "--gamma"), ::testing::Values(-2, -1.5, 0.0)));

class ParameterTrainCoef0 : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParameterTrainCoef0, parsing) {
    const auto &[flag, coef0] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, coef0));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(params.csvm_params.coef0.is_default());
    EXPECT_EQ(params.csvm_params.coef0, coef0);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainCoef0, ::testing::Combine(::testing::Values("-r", "--coef0"), ::testing::Values(-2, -1.5, -0.75, -0.001, 0, 0.001, 0.75, 1.5, 2)));

class ParameterTrainCost : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, int>> {};
TEST_P(ParameterTrainCost, parsing) {
    const auto &[flag, cost] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, cost));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(params.csvm_params.cost.is_default());
    EXPECT_EQ(params.csvm_params.cost, cost);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainCost, ::testing::Combine(::testing::Values("-c", "--cost"), ::testing::Range(-2, 3)));

class ParameterTrainEpsilon : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, double>> {};
TEST_P(ParameterTrainEpsilon, parsing) {
    const auto &[flag, eps] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, eps));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(params.epsilon.is_default());
    EXPECT_EQ(params.epsilon, eps);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainEpsilon, ::testing::Combine(::testing::Values("-e", "--epsilon"), ::testing::Values(10.0, 1.0, 0.0, 0.1, 0.01, 0.001, 0.0001, 0.00001)));

class ParameterTrainMaxIter : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::size_t>> {};
TEST_P(ParameterTrainMaxIter, parsing) {
    const auto &[flag, max_iter] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, max_iter));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_FALSE(params.max_iter.is_default());
    EXPECT_EQ(params.max_iter, max_iter);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainMaxIter, ::testing::Combine(::testing::Values("-i", "--max_iter"), ::testing::Values(1, 10, 100, 1000, 10000)));

class ParameterTrainMaxIterDeathTest : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, long long int>> {};
TEST_P(ParameterTrainMaxIterDeathTest, max_iter_explicit_less_or_equal_to_zero) {
    const auto &[flag, max_iter] = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, max_iter));
    // create parameter object
    EXPECT_DEATH((plssvm::detail::cmd::parameter_train{ this->argc, this->argv }), ::testing::HasSubstr(fmt::format("max_iter must be greater than 0, but is {}!", max_iter)));
}
INSTANTIATE_TEST_SUITE_P(ParameterTrainDeathTest, ParameterTrainMaxIterDeathTest, ::testing::Combine(::testing::Values("-i", "--max_iter"), ::testing::Values(-100, -10, -1, 0)));

class ParameterTrainBackend : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterTrainBackend, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to backend
    const auto backend = util::convert_from_string<plssvm::backend_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, value));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.backend, backend);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainBackend, ::testing::Combine(::testing::Values("-b", "--backend"), ::testing::Values("automatic", "OpenMP", "CUDA", "HIP", "OpenCL", "SYCL")));

class ParameterTrainTargetPlatform : public ParameterTrain, public ::testing::WithParamInterface<std::tuple<std::string, std::string>> {};
TEST_P(ParameterTrainTargetPlatform, parsing) {
    const auto &[flag, value] = GetParam();
    // convert string to target_platform
    const auto target_platform = util::convert_from_string<plssvm::target_platform>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} {} data.libsvm", flag, value));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.target, target_platform);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainTargetPlatform, ::testing::Combine(::testing::Values("-p", "--target_platform"), ::testing::Values("automatic", "cpu", "gpu_nvidia", "gpu_amd", "gpu_intel")));

#if defined(PLSSVM_HAS_SYCL_BACKEND)

class ParameterTrainSYCLKernelInvocation : public ParameterTrain, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterTrainSYCLKernelInvocation, parsing) {
    const std::string &value = GetParam();
    // convert string to sycl::kernel_invocation_type
    const auto sycl_kernel_invocation_type = util::convert_from_string<plssvm::sycl_generic::kernel_invocation_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict --sycl_kernel_invocation_type={} data.libsvm", value));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.sycl_kernel_invocation_type, sycl_kernel_invocation_type);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainSYCLKernelInvocation, ::testing::Values("automatic", "nd_range", "ND_RANGE", "hierarchical"));

class ParameterTrainSYCLImplementation : public ParameterTrain, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterTrainSYCLImplementation, parsing) {
    const std::string &value = GetParam();
    // convert string to sycl::implementation_type
    const auto sycl_implementation_type = util::convert_from_string<plssvm::sycl_generic::implementation_type>(value);
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-predict --sycl_implementation_type={} data.libsvm", value));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.sycl_implementation_type, sycl_implementation_type);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainSYCLImplementation, ::testing::Values("automatic", "hipSYCL", "DPCPP", "DPC++"));

#endif  // PLSSVM_HAS_SYCL_BACKEND

class ParameterTrainUseStringsAsLabels : public ParameterTrain, public ::testing::WithParamInterface<bool> {};
TEST_P(ParameterTrainUseStringsAsLabels, parsing) {
    const bool value = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train --use_strings_as_labels={} data.libsvm", value));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.strings_as_labels, value);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainUseStringsAsLabels, ::testing::Bool());

class ParameterTrainUseFloatAsRealType : public ParameterTrain, public ::testing::WithParamInterface<bool> {};
TEST_P(ParameterTrainUseFloatAsRealType, parsing) {
    const bool value = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train --use_float_as_real_type={} data.libsvm", value));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(params.float_as_real_type, value);
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainUseFloatAsRealType, ::testing::Bool());

class ParameterTrainQuiet : public ParameterTrain, public ::testing::WithParamInterface<std::string> {};
TEST_P(ParameterTrainQuiet, parsing) {
    const std::string &flag = GetParam();
    // create artificial command line arguments in test fixture
    this->CreateCMDArgs(fmt::format("./plssvm-train {} data.libsvm", flag));
    // create parameter object
    plssvm::detail::cmd::parameter_train params{ this->argc, this->argv };
    // test for correctness
    EXPECT_EQ(plssvm::verbose, flag.empty());
}
INSTANTIATE_TEST_SUITE_P(ParameterTrain, ParameterTrainQuiet, ::testing::Values("-q", "--quiet", ""));

// test whether nonsensical cmd arguments trigger the assertions
TEST_F(ParameterTrainDeathTest, too_few_argc) {
    EXPECT_DEATH((plssvm::detail::cmd::parameter_train{ 0, nullptr }), ::testing::HasSubstr("At least one argument is always given (the executable name), but argc is 0!"));
}
TEST_F(ParameterTrainDeathTest, nullptr_argv) {
    EXPECT_DEATH((plssvm::detail::cmd::parameter_train{ 1, nullptr }), ::testing::HasSubstr("At least one argument is always given (the executable name), but argv is a nullptr!"));
}

TEST_F(ParameterTrainDeathTest, help) {
    this->CreateCMDArgs("./plssvm-train --help");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_train{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
TEST_F(ParameterTrainDeathTest, version) {
    this->CreateCMDArgs("./plssvm-train --version");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_train{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_SUCCESS), "");
}
TEST_F(ParameterTrainDeathTest, no_positional_argument) {
    this->CreateCMDArgs("./plssvm-train");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_train{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_FAILURE), "Error missing input file!\n");
}

TEST_F(ParameterTrainDeathTest, too_many_positional_arguments) {
    this->CreateCMDArgs("./plssvm-train p1 p2 p3 p4");
    EXPECT_EXIT((plssvm::detail::cmd::parameter_train{ this->argc, this->argv }), ::testing::ExitedWithCode(EXIT_FAILURE), ::testing::HasSubstr("Only up to two positional options may be given, but 2 (\"p3 p4\") additional option(s) where provided!"));
}

TEST_F(ParameterTrainDeathTest, unrecognized_option) {
    this->CreateCMDArgs("./plssvm-train --foo bar");
    EXPECT_DEATH((plssvm::detail::cmd::parameter_train{ this->argc, this->argv }), "");
}