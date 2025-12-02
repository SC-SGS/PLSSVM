/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the base C-SVM functions through its mock class.
 */

#include "plssvm/svm/csvm.hpp"

#include "plssvm/backend_types.hpp"                   // plssvm::csvm_backend_exists, plssvm::csvm_backend_exists_v, plssvm::backend_csvm_type, plssvm::backend_csvm_type_t
#include "plssvm/backends/SYCL/detail/constants.hpp"  // NOLINT: namespace plssvm::sycl
#include "plssvm/constants.hpp"                       // plssvm::real_type
#include "plssvm/core.hpp"                            // NOLINT: include all csvm_backend_exists_v specializations
#include "plssvm/kernel_function_types.hpp"           // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                       // plssvm::parameter
#include "plssvm/target_platforms.hpp"                // plssvm::target_platform

#include "tests/svm/mock_csvm.hpp"  // mock_csvm
#include "tests/utility.hpp"        // util::redirect_output

#include "gmock/gmock.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_THAT

#include <iostream>  // std::clog

class BaseCSVM : public ::testing::Test { };

TEST(BaseCSVM, DefaultConstructFromParameter) {
    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{};

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), plssvm::parameter{});
}

TEST(BaseCSVM, ConstructFromParameter) {
    // create parameter
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.01 } };

    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{ params };

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}

TEST(BaseCSVM, ConstructLinearFromNamedParameters) {
    // correct parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 };

    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{ plssvm::kernel_type = params.kernel_type, plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}

TEST(BaseCSVM, ConstructPolynomialFromNamedParameters) {
    // correct parameter
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.1 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 0.001 } };

    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{ plssvm::kernel_type = params.kernel_type,
                          plssvm::degree = params.degree,
                          plssvm::gamma = params.gamma,
                          plssvm::coef0 = params.coef0,
                          plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}

TEST(BaseCSVM, ConstructRadialBasisFunctionFromNamedParameters) {
    // correct parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.00001, plssvm::cost = 10.0 };

    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{ plssvm::kernel_type = params.kernel_type,
                          plssvm::gamma = params.gamma,
                          plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}

TEST(BaseCSVM, ConstructSigmoidFromNamedParameters) {
    // correct parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::sigmoid, plssvm::gamma = 0.00001, plssvm::cost = 10.0 };

    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{ plssvm::kernel_type = params.kernel_type,
                          plssvm::gamma = params.gamma,
                          plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}

TEST(BaseCSVM, ConstructLaplacianFromNamedParameters) {
    // correct parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::laplacian, plssvm::gamma = 0.00001, plssvm::cost = 10.0 };

    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{ plssvm::kernel_type = params.kernel_type,
                          plssvm::gamma = params.gamma,
                          plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}

TEST(BaseCSVM, ConstructChiSquaredFromNamedParameters) {
    // correct parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::chi_squared, plssvm::gamma = 0.00001, plssvm::cost = 10.0 };

    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{ plssvm::kernel_type = params.kernel_type,
                          plssvm::gamma = params.gamma,
                          plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}

TEST(BaseCSVM, GetTargetPlatforms) {
    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{};

    EXPECT_EQ(csvm.get_target_platform(), plssvm::target_platform::automatic);
}

TEST(BaseCSVM, GetParams) {
    // create parameter
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.01 } };

    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    const mock_csvm csvm{ params };

    // check whether the parameters have been set correctly
    const plssvm::parameter csvm_params = csvm.get_params();
    EXPECT_EQ(csvm_params, params);
    EXPECT_TRUE(csvm_params.equivalent(params));
}

TEST(BaseCSVM, SetParamsFromParameter) {
    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    mock_csvm csvm{};
    ASSERT_EQ(csvm.get_params(), plssvm::parameter{});

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.01 } };

    // set csvm parameter to new values
    csvm.set_params(params);

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}

TEST(BaseCSVM, SetParamsFromNamedParameters) {
    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    mock_csvm csvm{};
    ASSERT_EQ(csvm.get_params(), plssvm::parameter{});

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.01 } };

    // set csvm parameter to new values
    csvm.set_params(plssvm::kernel_type = plssvm::kernel_function_type::polynomial,
                    plssvm::degree = 4,
                    plssvm::gamma = 0.2,
                    plssvm::coef0 = 0.1,
                    plssvm::cost = 0.01);

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}

TEST(BaseCSVM, CsvmBackendExists) {
    // test whether the given C-SVM backend exist
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::openmp::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::openmp::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::openmp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::openmp::csvm>::value);
#endif

#if defined(PLSSVM_HAS_HPX_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::hpx::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::hpx::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::hpx::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::hpx::csvm>::value);
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::cuda::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::cuda::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::cuda::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::cuda::csvm>::value);
#endif

#if defined(PLSSVM_HAS_HIP_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::hip::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::hip::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::hip::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::hip::csvm>::value);
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::opencl::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::opencl::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::opencl::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::opencl::csvm>::value);
#endif

#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::sycl::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::sycl::csvm>::value);
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvm>::value);
    #else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvm>::value);
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvm>::value);
    #else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvm>::value);
    #endif
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::sycl::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::sycl::csvm>::value);
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvm>::value);
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvm>::value);
#endif

#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::kokkos::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::kokkos::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::kokkos::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::kokkos::csvm>::value);
#endif
}

class BaseCSVMWarning : public BaseCSVM,
                        protected util::redirect_output<&std::clog> { };

TEST_F(BaseCSVMWarning, ConstructUnusedParameterWarningDegree) {
    // start capture of std::clog
    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    [[maybe_unused]] const mock_csvm csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::degree = 2 };
    // end capture of std::clog

    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("WARNING: degree parameter provided, which is not used in the linear kernel (u'*v)!"));
}

TEST_F(BaseCSVMWarning, ConstructUnusedParameterWarningGamma) {
    // start capture of std::clog
    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    [[maybe_unused]] const mock_csvm csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::gamma = 0.1 };
    // end capture of std::clog

    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("WARNING: gamma parameter provided, which is not used in the linear kernel (u'*v)!"));
}

TEST_F(BaseCSVMWarning, ConstructUnusedParameterWarningCoef0) {
    // start capture of std::clog
    // create C-SVM: must be done using the mock class since the csvm base class is pure virtual
    [[maybe_unused]] const mock_csvm csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::coef0 = 0.1 };
    // end capture of std::clog

    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("WARNING: coef0 parameter provided, which is not used in the linear kernel (u'*v)!"));
}
