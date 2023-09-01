/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functions performing the actual kernel calculations using the OpenMP backend.
 */

#include "plssvm/backends/OpenMP/cg_explicit/kernel_matrix_assembly.hpp"

#include "plssvm/constants.hpp"  // plssvm::real_type

#include "../../custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_VECTOR_NEAR

#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, EXPECT_DEATH

#include <vector>  // std::vector

TEST(OpenMPCSVMKernelMatrixAssembly, linear_assembly) {
    using plssvm::real_type;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const plssvm::aos_matrix<plssvm::real_type> data{ std::vector<std::vector<real_type>>{
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } } } };

    const std::vector<real_type> q = { real_type{ 1.0 }, real_type{ 2.0 } };
    const real_type QA_cost{ 0.5 };
    const real_type cost{ 1.0 };

#if defined(PLSSVM_USE_GEMM)
    const std::vector<real_type> correct_ret = {
        real_type{ 0.5 }, real_type{ 0.5 }, real_type{ 0.5 }, real_type{ 10.5 }
    };
#else
    const std::vector<real_type> correct_ret = {
        real_type{ 0.5 }, real_type{ 0.5 }, real_type{ 10.5 }
    };
#endif
    std::vector<real_type> ret(correct_ret.size());

    // assemble kernel matrix
    plssvm::openmp::device_kernel_assembly_linear(q, ret, data, QA_cost, cost);

    EXPECT_FLOATING_POINT_VECTOR_NEAR(ret, correct_ret);
}
TEST(OpenMPCSVMKernelMatrixAssembly, polynomial_assembly) {
    using plssvm::real_type;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const plssvm::aos_matrix<plssvm::real_type> data{ std::vector<std::vector<real_type>>{
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } } } };

    const std::vector<real_type> q = { real_type{ 1.0 }, real_type{ 2.0 } };
    const real_type QA_cost{ 0.5 };
    const real_type cost{ 1.0 };

#if defined(PLSSVM_USE_GEMM)
    const std::vector<real_type> correct_ret = {
        real_type{ 3.91 }, real_type{ 2.79 }, real_type{ 2.79 }, real_type{ 8.39 }
    };
#else
    const std::vector<real_type> correct_ret = {
        real_type{ 3.91 }, real_type{ 2.79 }, real_type{ 8.39 }
    };
#endif
    std::vector<real_type> ret(correct_ret.size());

    // assemble kernel matrix
    plssvm::openmp::device_kernel_assembly_polynomial(q, ret, data, QA_cost, cost, 2, real_type{ 0.1 }, real_type{ 2.0 });

    EXPECT_FLOATING_POINT_VECTOR_NEAR(ret, correct_ret);
}
TEST(OpenMPCSVMKernelMatrixAssembly, rbf_assembly) {
    using plssvm::real_type;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const plssvm::aos_matrix<plssvm::real_type> data{ std::vector<std::vector<real_type>>{
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } } } };

    const std::vector<real_type> q = { real_type{ 1.0 }, real_type{ 2.0 } };
    const real_type QA_cost{ 0.5 };
    const real_type cost{ 1.0 };

    // assemble kernel matrix
#if defined(PLSSVM_USE_GEMM)
    const std::vector<real_type> correct_ret = {
        real_type{ 0.5 }, std::exp(real_type{ -0.8 }) + real_type{ -2.5 }, std::exp(real_type{ -0.8 }) + real_type{ -2.5 }, real_type{ -1.5 }
    };
#else
    const std::vector<real_type> correct_ret = {
        real_type{ 0.5 }, std::exp(real_type{ -0.8 }) + real_type{ -2.5 }, real_type{ -1.5 }
    };
#endif
    std::vector<real_type> ret(correct_ret.size());

    plssvm::openmp::device_kernel_assembly_rbf(q, ret, data, QA_cost, cost, real_type{ 0.1 });

    EXPECT_FLOATING_POINT_VECTOR_NEAR(ret, correct_ret);
}

TEST(OpenMPCSVMKernelMatrixAssemblyDeathTest, linear_assembly) {
    using plssvm::real_type;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const plssvm::aos_matrix<plssvm::real_type> data{ std::vector<std::vector<real_type>>{
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } } } };
    const real_type QA_cost{ 0.5 };
    const real_type cost{ 1.0 };

    const std::vector<real_type> q(data.num_rows() - 1);
#if defined(PLSSVM_USE_GEMM)
    std::vector<real_type> correct_ret((data.num_rows() - 1) * (data.num_rows() - 1));
#else
    std::vector<real_type> correct_ret((data.num_rows() - 1) * (data.num_rows()) / 2);
#endif

    EXPECT_DEATH(plssvm::openmp::device_kernel_assembly_linear(std::vector<real_type>(1), correct_ret, data, QA_cost, cost), ::testing::HasSubstr("Sizes mismatch!: 1 != 2"));
    std::vector<real_type> ret(1);
#if defined(PLSSVM_USE_GEMM)
    EXPECT_DEATH(plssvm::openmp::device_kernel_assembly_linear(q, ret, data, QA_cost, cost), ::testing::HasSubstr("Sizes mismatch (GEMM)!: 1 != 4"));
#else
    EXPECT_DEATH(plssvm::openmp::device_kernel_assembly_linear(q, ret, data, QA_cost, cost), ::testing::HasSubstr("Sizes mismatch (SYMM)!: 1 != 3"));
#endif
    EXPECT_DEATH(plssvm::openmp::device_kernel_assembly_linear(q, correct_ret, data, QA_cost, real_type{ 0.0 }), ::testing::HasSubstr("cost must not be 0.0 since it is 1 / plssvm::cost!"));
}
TEST(OpenMPCSVMKernelMatrixAssemblyDeathTest, polynomial_assembly) {
    using plssvm::real_type;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const plssvm::aos_matrix<plssvm::real_type> data{ std::vector<std::vector<real_type>>{
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } } } };
    const real_type QA_cost{ 0.5 };
    const real_type cost{ 1.0 };

    const std::vector<real_type> q(data.num_rows() - 1);
#if defined(PLSSVM_USE_GEMM)
    std::vector<real_type> ret((data.num_rows() - 1) * (data.num_rows() - 1));
#else
    std::vector<real_type> ret((data.num_rows() - 1) * (data.num_rows()) / 2);
#endif
    EXPECT_DEATH(plssvm::openmp::device_kernel_assembly_polynomial(q, ret, data, QA_cost, cost, 2, real_type{ 0.0 }, real_type{ 1.0 }), ::testing::HasSubstr("gamma must be greater than 0, but is 0!"));
}
TEST(OpenMPCSVMKernelMatrixAssemblyDeathTest, rbf_assembly) {
    using plssvm::real_type;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const plssvm::aos_matrix<plssvm::real_type> data{ std::vector<std::vector<real_type>>{
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } } } };
    const real_type QA_cost{ 0.5 };
    const real_type cost{ 1.0 };

    const std::vector<real_type> q(data.num_rows() - 1);
#if defined(PLSSVM_USE_GEMM)
    std::vector<real_type> ret((data.num_rows() - 1) * (data.num_rows() - 1));
#else
    std::vector<real_type> ret((data.num_rows() - 1) * (data.num_rows()) / 2);
#endif
    EXPECT_DEATH(plssvm::openmp::device_kernel_assembly_rbf(q, ret, data, QA_cost, cost, real_type{ 0.0 }), ::testing::HasSubstr("gamma must be greater than 0, but is 0!"));
}