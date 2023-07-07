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

#include "../../custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_2D_VECTOR_NEAR
#include "../../naming.hpp"              // naming::real_type_to_name
#include "../../types_to_test.hpp"       // util::real_type_gtest

#include "gmock/gmock-matchers.h"        // ::testing::HasSubstr
#include "gtest/gtest.h"                 // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_DEATH, ::testing::Test

#include <vector>                        // std::vector

template <typename T>
class OpenMPSVMKernelMatrixAssembly : public ::testing::Test {};
TYPED_TEST_SUITE(OpenMPSVMKernelMatrixAssembly, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(OpenMPSVMKernelMatrixAssembly, linear_assembly) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> correct_data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    const std::vector<std::vector<real_type>> correct_ret = {
        { real_type{ 0.5 }, real_type{ 0.5 } },
        { real_type{ 0.5 }, real_type{ 10.5 } },
    };

    const std::vector<real_type> q = { real_type{ 1.0 }, real_type{ 2.0 } };
    const real_type QA_cost{ 0.5 };
    const real_type cost{ 1.0 };

    // assemble kernel matrix
    std::vector<std::vector<real_type>> ret(correct_data.size() - 1, std::vector<real_type>(correct_data.size() - 1, real_type{ 0.0 }));
    plssvm::openmp::linear_kernel_matrix_assembly(q, ret, correct_data, QA_cost, cost);

    EXPECT_FLOATING_POINT_2D_VECTOR_NEAR(ret, correct_ret);
}
TYPED_TEST(OpenMPSVMKernelMatrixAssembly, polynomial_assembly) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> correct_data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    const std::vector<std::vector<real_type>> correct_ret = {
        { real_type{ 3.91 }, real_type{ 2.79 } },
        { real_type{ 2.79 }, real_type{ 8.39 } },
    };

    const std::vector<real_type> q = { real_type{ 1.0 }, real_type{ 2.0 } };
    const real_type QA_cost{ 0.5 };
    const real_type cost{ 1.0 };

    // assemble kernel matrix
    std::vector<std::vector<real_type>> ret(correct_data.size() - 1, std::vector<real_type>(correct_data.size() - 1, real_type{ 0.0 }));
    plssvm::openmp::polynomial_kernel_matrix_assembly(q, ret, correct_data, QA_cost, cost, 2, real_type{ 0.1 }, real_type{ 2.0 });

    EXPECT_FLOATING_POINT_2D_VECTOR_NEAR(ret, correct_ret);
}
TYPED_TEST(OpenMPSVMKernelMatrixAssembly, rbf_assembly) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> correct_data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    const std::vector<std::vector<real_type>> correct_ret = {
        { real_type{ 0.5 }, std::exp(real_type{ -0.8 }) + real_type{ -2.5 } },
        { std::exp(real_type{ -0.8 }) + real_type{ -2.5 }, real_type{ -1.5 } },
    };

    const std::vector<real_type> q = { real_type{ 1.0 }, real_type{ 2.0 } };
    const real_type QA_cost{ 0.5 };
    const real_type cost{ 1.0 };

    // assemble kernel matrix
    std::vector<std::vector<real_type>> ret(correct_data.size() - 1, std::vector<real_type>(correct_data.size() - 1, real_type{ 0.0 }));
    plssvm::openmp::rbf_kernel_matrix_assembly(q, ret, correct_data, QA_cost, cost, real_type{ 0.1 });

    EXPECT_FLOATING_POINT_2D_VECTOR_NEAR(ret, correct_ret);
}

template <typename T>
class OpenMPSVMKernelMatrixAssemblyDeathTest : public ::testing::Test {};
TYPED_TEST_SUITE(OpenMPSVMKernelMatrixAssemblyDeathTest, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(OpenMPSVMKernelMatrixAssemblyDeathTest, linear_assembly) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> correct_data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    const std::vector<real_type> correct_q(correct_data.size() - 1);
    std::vector<std::vector<real_type>> correct_ret(correct_data.size() - 1, std::vector<real_type>(correct_data.size() - 1));
    const std::vector<real_type> correct_d(correct_data.size() - 1);

    EXPECT_DEATH(plssvm::openmp::linear_kernel_matrix_assembly(std::vector<real_type>(1), correct_ret, correct_data, real_type{ 0.0 }, real_type{ 1.0 }), ::testing::HasSubstr("Sizes mismatch!: 1 != 2"));
    std::vector<std::vector<real_type>> ret(1);
    EXPECT_DEATH(plssvm::openmp::linear_kernel_matrix_assembly(correct_q, ret, correct_data, real_type{ 0.0 }, real_type{ 1.0 }), ::testing::HasSubstr("Sizes mismatch!: 2 != 1"));
    EXPECT_DEATH(plssvm::openmp::linear_kernel_matrix_assembly(correct_q, correct_ret, correct_data, real_type{ 0.0 }, real_type{ 0.0 }), ::testing::HasSubstr("cost must not be 0.0 since it is 1 / plssvm::cost!"));
}
TYPED_TEST(OpenMPSVMKernelMatrixAssemblyDeathTest, polynomial_assembly) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    const std::vector<real_type> q(data.size() - 1);
    std::vector<std::vector<real_type>> ret(data.size() - 1, std::vector<real_type>(data.size() - 1));
    const real_type QA_cost{};
    const real_type cost{ 1.0 };
    EXPECT_DEATH(plssvm::openmp::polynomial_kernel_matrix_assembly(q, ret, data, QA_cost, cost, 2, real_type{ 0.0 }, real_type{ 1.0 }), ::testing::HasSubstr("gamma must be greater than 0, but is 0!"));
}
TYPED_TEST(OpenMPSVMKernelMatrixAssemblyDeathTest, rbf_assembly) {
    using real_type = TypeParam;

    // create vectors with mismatching sizes: note that the provided out vector is one smaller than the data vector!
    const std::vector<std::vector<real_type>> data = {
        { real_type{ 0.0 }, real_type{ 1.0 } },
        { real_type{ 2.0 }, real_type{ 3.0 } },
        { real_type{ 4.0 }, real_type{ 5.0 } }
    };
    const std::vector<real_type> q(data.size() - 1);
    std::vector<std::vector<real_type>> ret(data.size() - 1, std::vector<real_type>(data.size() - 1));

    EXPECT_DEATH(plssvm::openmp::rbf_kernel_matrix_assembly(q, ret, data, real_type{ 0.0 }, real_type{ 1.0 }, real_type{ 0.0 }), ::testing::HasSubstr("gamma must be greater than 0, but is 0!"));
}