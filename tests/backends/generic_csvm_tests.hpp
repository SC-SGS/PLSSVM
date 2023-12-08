/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic tests for all backends to reduce code duplication.
 */

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_CSVM_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_CSVM_TESTS_HPP_
#pragma once

#include "plssvm/classification_types.hpp"      // plssvm::classification_type
#include "plssvm/constants.hpp"                 // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/data_set.hpp"                  // plssvm::data_set
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/memory_size.hpp"        // memory size literals
#include "plssvm/detail/simple_any.hpp"         // plssvm::detail::simple_any
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/kernel_function_types.hpp"     // plssvm::csvm_to_backend_type_v, plssvm::backend_type
#include "plssvm/matrix.hpp"                    // plssvm::aos_matrix, plssvm::layout_type
#include "plssvm/model.hpp"                     // plssvm::model
#include "plssvm/parameter.hpp"                 // plssvm::parameter
#include "plssvm/solver_types.hpp"              // plssvm::solver_type
#include "plssvm/target_platforms.hpp"          // plssvm::target_platform

#include "backends/compare.hpp"    // compare::{kernel_function, perform_dimensional_reduction}
#include "custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_MATRIX_EQ, EXPECT_FLOATING_POINT_VECTOR_NEAR, EXPECT_FLOATING_POINT_NEAR
#include "types_to_test.hpp"       // util::{test_parameter_type_at_t, test_parameter_value_at_v}
#include "utility.hpp"             // util::{redirect_output, generate_specific_matrix, construct_from_tuple, flatten, generate_random_matrix}

#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads
#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_EQ, EXPECT_NE, EXPECT_GT, EXPECT_TRUE, EXPECT_DEATH,
                          // ASSERT_EQ, GTEST_SKIP, SUCCEED, ::testing::Test

#include <array>    // std::array
#include <cmath>    // std::sqrt, std::abs, std::exp, std::pow
#include <cstddef>  // std::size_t
#include <limits>   // std::numeric_limits::epsilon
#include <tuple>    // std::ignore
#include <utility>  // std::move
#include <vector>   // std::vector

namespace util {

/**
 * @brief Add padding to the @p matr of @p shape. Respects whether `PLSSVM_USE_GEMM` is set to `ON` or `OFF`.
 * @details Adds no padding entries for the OpenMP backend.
 * @tparam T the type of the values in the matrix
 * @param[in] matr the matrix to add padding
 * @param[in] shape the shape of the underlying matrix represented by the 1D @p matr
 * @return the padded matrix (`[[nodiscard]]`)
 */
template <typename plssvm_csvm_type, typename T>
[[nodiscard]] inline std::vector<T> pad_1D_vector(std::vector<T> matr, [[maybe_unused]] const std::array<std::size_t, 2> shape) {
    // the OpenMP backend has no padding!
    if constexpr (plssvm::csvm_to_backend_type_v<plssvm_csvm_type> != plssvm::backend_type::openmp) {
        std::size_t idx{ matr.size() };
        for (std::size_t i = 0; i < shape[0]; ++i) {
            matr.insert(matr.cbegin() + idx, plssvm::PADDING_SIZE, plssvm::real_type{ 0.0 });
#if defined(PLSSVM_USE_GEMM)
            idx -= shape[1];
#else
            idx -= i + 1;
#endif
        }
#if defined(PLSSVM_USE_GEMM)
        matr.insert(matr.cend(), plssvm::PADDING_SIZE * (shape[1] + plssvm::PADDING_SIZE), plssvm::real_type{ 0.0 });
#else
        matr.insert(matr.cend(), plssvm::PADDING_SIZE * (plssvm::PADDING_SIZE + 1) / 2, plssvm::real_type{ 0.0 });
#endif
    }
    return matr;
}

/**
 * @brief Initialize the matrix adding necessary padding entries using the values provided by @p matr based on the @p csvm_type.
 * @details Returns a `plssvm::detail::simple_any` with a value based on the @p csvm_type.
 * @tparam plssvm_csvm_type the PLSSVM CSVM backend type
 * @tparam device_ptr_type the device pointer type; only used if a GPU backend is used
 * @tparam T the type of the values in the matrix
 * @tparam used_csvm_type the type of the @p csvm; may also be a Mock CSVM!
 * @param[in] matr the matrix
 * @param[in] csvm the CSVM encapsulating the device on which the matrix should be allocated
 * @param[in] shape the shape of the underlying matrix represented by the 1D @p matr
 * @return a `plssvm::detail::simple_any` with a wrapped value usable in the PLSSVM functions (`[[nodiscard]]`)
 */
template <typename plssvm_csvm_type, typename device_ptr_type, typename T, typename used_csvm_type>
[[nodiscard]] inline auto init_matrix(std::vector<T> matr, [[maybe_unused]] used_csvm_type &csvm, [[maybe_unused]] const std::array<std::size_t, 2> shape) {
    // create device pointer
    if constexpr (plssvm::csvm_to_backend_type_v<plssvm_csvm_type> == plssvm::backend_type::openmp) {
        return plssvm::detail::simple_any{ std::move(matr) };
    } else {
        // add padding
        matr = pad_1D_vector<plssvm_csvm_type>(std::move(matr), shape);

        device_ptr_type ptr{ matr.size(), csvm.devices_[0] };
        ptr.copy_to_device(matr);
        return plssvm::detail::simple_any{ std::move(ptr) };
    }
}

}  // namespace util

//*************************************************************************************************************************************//
//                                                   CSVM tests depending on nothing                                                   //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVM : public ::testing::Test, protected util::redirect_output<> {};
TYPED_TEST_SUITE_P(GenericCSVM);

TYPED_TEST_P(GenericCSVM, move_constructor) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;

    // create default CSVM
    csvm_type svm{};

    // get current state
    const plssvm::parameter params = svm.get_params();
    const plssvm::target_platform target = svm.get_target_platform();

    // move construct new CSVM
    const csvm_type new_svm{ std::move(svm) };

    // check that the state of the newly constructed CSVM matches the old state of the moved-from CSVM
    EXPECT_EQ(new_svm.get_params(), params);
    EXPECT_EQ(new_svm.get_target_platform(), target);
}
TYPED_TEST_P(GenericCSVM, move_assignment) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;

    // create default CSVM
    csvm_type svm{};

    // get current state
    const plssvm::parameter params = svm.get_params();
    const plssvm::target_platform target = svm.get_target_platform();

    // construct new CSVM with a non-default state
    csvm_type new_svm{ plssvm::parameter{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial } };

    // move assign old CSVM to the new one
    new_svm = std::move(svm);

    // check that the state of the newly constructed CSVM matches the old state of the moved-from CSVM
    EXPECT_EQ(new_svm.get_params(), params);
    EXPECT_EQ(new_svm.get_target_platform(), target);
}

TYPED_TEST_P(GenericCSVM, get_target_platform) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;

    // create C-SVM
    const csvm_type svm = util::construct_from_tuple<csvm_type>(csvm_test_type::additional_arguments);

    // after construction: get_target_platform must refer to a plssvm::target_platform that is not automatic
    EXPECT_NE(svm.get_target_platform(), plssvm::target_platform::automatic);
}
TYPED_TEST_P(GenericCSVM, get_device_memory) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the available device memory should be greater than 0!
    using namespace plssvm::detail::literals;
    EXPECT_GT(svm.get_device_memory(), 0_B);
}
TYPED_TEST_P(GenericCSVM, get_max_mem_alloc_size) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the maximum memory allocation size should be greater than 0!
    using namespace plssvm::detail::literals;
    EXPECT_GT(svm.get_max_mem_alloc_size(), 0_B);
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVM,
                            move_constructor, move_assignment,
                            get_target_platform, get_device_memory, get_max_mem_alloc_size);
// clang-format on

//*************************************************************************************************************************************//
//                                               CSVM tests depending on the solver type                                               //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMSolver : public GenericCSVM<T> {};
TYPED_TEST_SUITE_P(GenericCSVMSolver);

TYPED_TEST_P(GenericCSVMSolver, setup_data_on_devices) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    // minimal example
    const auto input = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(3, 3, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::setup_data_on_device is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ASSERT_ENABLED)
        EXPECT_DEATH(std::ignore = svm.setup_data_on_devices(solver, input), "An explicit solver type must be provided instead of solver_type::automatic!");
#else
        SUCCEED() << "Solver type is automatic, but assertions are disabled!";
#endif
    } else {
        // perform data setup on the device
        const plssvm::detail::simple_any s_data_d = svm.setup_data_on_devices(solver, input);
        const auto &data_d = s_data_d.get<device_ptr_type>();

        if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp) {
            // OpenMP special case
            ASSERT_EQ(data_d->shape(), (std::array<std::size_t, 2>{ 3, 3 }));
            ASSERT_EQ(data_d->num_entries(), 9);
            ASSERT_TRUE(data_d->is_padded());
            ASSERT_EQ(data_d->padding()[0], plssvm::PADDING_SIZE);
            ASSERT_EQ(data_d->padding()[1], plssvm::PADDING_SIZE);
            EXPECT_FLOATING_POINT_MATRIX_EQ(*data_d, input);
        } else {
            // generic GPU case
            ASSERT_EQ(data_d.extents(), (std::array<std::size_t, 2>{ 3, 3 }));
            ASSERT_EQ(data_d.size(), (3 + plssvm::PADDING_SIZE) * (3 + plssvm::PADDING_SIZE));
            ASSERT_EQ(data_d.padding(0), plssvm::PADDING_SIZE);
            ASSERT_EQ(data_d.padding(1), plssvm::PADDING_SIZE);
            std::vector<plssvm::real_type> data(data_d.size());
            data_d.copy_to_host(data);
            const std::vector<plssvm::real_type> correct_data{ input.data(), input.data() + input.num_entries_padded() };
            EXPECT_FLOATING_POINT_VECTOR_EQ(data, correct_data);
        }
    }
}

TYPED_TEST_P(GenericCSVMSolver, blas_level_3_without_C) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::blas_level_3 is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    [[maybe_unused]] const plssvm::real_type alpha{ 1.0 };

#if defined(PLSSVM_USE_GEMM)
    // clang-format off
    std::vector<plssvm::real_type> matr_A{
        plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 },
        plssvm::real_type{ 0.2 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 },
        plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 }
    };
        // clang-format on
#else
    std::vector<plssvm::real_type> matr_A = { plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 } };
#endif

    const plssvm::detail::simple_any A{ util::init_matrix<csvm_type, device_ptr_type>(matr_A, svm, { 3, 3 }) };

    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } },
                                                   plssvm::PADDING_SIZE,
                                                   plssvm::PADDING_SIZE };

    [[maybe_unused]] const plssvm::real_type beta{ 0.0 };
    plssvm::soa_matrix<plssvm::real_type> C{ 3, 3, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE };
    plssvm::soa_matrix<plssvm::real_type> C2{ C };

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ASSERT_ENABLED)
        EXPECT_DEATH(svm.blas_level_3(solver, alpha, A, B, beta, C), "An explicit solver type must be provided instead of solver_type::automatic!");
#else
        SUCCEED() << "Solver type is automatic, but assertions are disabled!";
#endif
    } else {
        // perform BLAS calculation
        svm.blas_level_3(solver, alpha, A, B, beta, C);

        // check C for correctness
        const plssvm::soa_matrix<plssvm::real_type> correct_C{ { { plssvm::real_type{ 1.4 }, plssvm::real_type{ 6.5 }, plssvm::real_type{ 9.8 } },
                                                                 { plssvm::real_type{ 3.2 }, plssvm::real_type{ 14.6 }, plssvm::real_type{ 21.5 } },
                                                                 { plssvm::real_type{ 5.0 }, plssvm::real_type{ 22.7 }, plssvm::real_type{ 33.2 } } },
                                                               plssvm::PADDING_SIZE,
                                                               plssvm::PADDING_SIZE };
        EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);

        // check wrapper function
        std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);
        EXPECT_EQ(C2, C);
    }
}
TYPED_TEST_P(GenericCSVMSolver, blas_level_3) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::blas_level_3 is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    [[maybe_unused]] const plssvm::real_type alpha{ 1.0 };

#if defined(PLSSVM_USE_GEMM)
    // clang-format off
    std::vector<plssvm::real_type> matr_A{
        plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 },
        plssvm::real_type{ 0.2 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 },
        plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 }
    };
        // clang-format on
#else
    std::vector<plssvm::real_type> matr_A = { plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 } };
#endif
    const plssvm::detail::simple_any A{ util::init_matrix<csvm_type, device_ptr_type>(matr_A, svm, { 3, 3 }) };

    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } },
                                                   plssvm::PADDING_SIZE,
                                                   plssvm::PADDING_SIZE };

    [[maybe_unused]] const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(3, 3, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    plssvm::soa_matrix<plssvm::real_type> C2{ C };

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ASSERT_ENABLED)
        EXPECT_DEATH(svm.blas_level_3(solver, alpha, A, B, beta, C), "An explicit solver type must be provided instead of solver_type::automatic!");
#else
        SUCCEED() << "Solver type is automatic, but assertions are disabled!";
#endif
    } else {
        // perform BLAS calculation
        svm.blas_level_3(solver, alpha, A, B, beta, C);

        // check C for correctness
        const plssvm::soa_matrix<plssvm::real_type> correct_C{ { { plssvm::real_type{ 1.45 }, plssvm::real_type{ 6.6 }, plssvm::real_type{ 9.95 } },
                                                                 { plssvm::real_type{ 3.75 }, plssvm::real_type{ 15.2 }, plssvm::real_type{ 22.15 } },
                                                                 { plssvm::real_type{ 6.05 }, plssvm::real_type{ 23.8 }, plssvm::real_type{ 34.35 } } },
                                                               plssvm::PADDING_SIZE,
                                                               plssvm::PADDING_SIZE };
        EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);

        // check wrapper function
        std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);
        EXPECT_EQ(C2, C);
    }
}

TYPED_TEST_P(GenericCSVMSolver, conjugate_gradients_trivial) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::conjugate_gradients is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // create the data that should be used
#if defined(PLSSVM_USE_GEMM)
    // clang-format off
    const std::vector<plssvm::real_type> matr_A{
        plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 },
        plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 },
        plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 },
        plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }
    };
        // clang-format on
#else
    const std::vector<plssvm::real_type> matr_A{ plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } };
#endif
    const plssvm::detail::simple_any A{ util::init_matrix<csvm_type, device_ptr_type>(matr_A, svm, { 4, 4 }) };
    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                     { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } } };

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ASSERT_ENABLED)
        EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.00001 }, 4, solver), "An explicit solver type must be provided instead of solver_type::automatic!");
#else
        SUCCEED() << "Solver type is automatic, but assertions are disabled!";
#endif
    } else {
        // solve AX = B
        const auto [X, num_iter] = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.00001 }, 4, solver);

        // check result
        EXPECT_FLOATING_POINT_MATRIX_NEAR(X, (plssvm::soa_matrix<plssvm::real_type>{ B, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
        EXPECT_GT(num_iter, 0);
    }
}
TYPED_TEST_P(GenericCSVMSolver, conjugate_gradients) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::conjugate_gradients is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // create the data that should be used
#if defined(PLSSVM_USE_GEMM)
    // clang-format off
    const std::vector<plssvm::real_type> matr_A{
        plssvm::real_type{ 4.0 }, plssvm::real_type{ 1.0 },
        plssvm::real_type{ 1.0 }, plssvm::real_type{ 3.0 }
    };
        // clang-format on
#else
    const std::vector<plssvm::real_type> matr_A{ plssvm::real_type{ 4.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 3.0 } };
#endif
    const plssvm::detail::simple_any A{ util::init_matrix<csvm_type, device_ptr_type>(matr_A, svm, { 2, 2 }) };
    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 } },
                                                     { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 } } } };
    const plssvm::soa_matrix<plssvm::real_type> correct_X{ { { plssvm::real_type{ 1.0 / 11.0 }, plssvm::real_type{ 7.0 / 11.0 } },
                                                             { plssvm::real_type{ 1.0 / 11.0 }, plssvm::real_type{ 7.0 / 11.0 } } },
                                                           plssvm::PADDING_SIZE,
                                                           plssvm::PADDING_SIZE };

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ASSERT_ENABLED)
        EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.00001 }, 2, solver), "An explicit solver type must be provided instead of solver_type::automatic!");
#else
        SUCCEED() << "Solver type is automatic, but assertions are disabled!";
#endif
    } else {
        // solve AX = B
        const auto [X, num_iters] = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.00001 }, 2, solver);

        // check result
        EXPECT_FLOATING_POINT_MATRIX_NEAR(X, correct_X);
        EXPECT_GT(num_iters, 0);
    }
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolver,
                            setup_data_on_devices,
                            blas_level_3_without_C, blas_level_3,
                            conjugate_gradients_trivial, conjugate_gradients);
// clang-format on

//*************************************************************************************************************************************//
//                                           CSVM tests depending on the kernel function type                                          //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMKernelFunction : public GenericCSVM<T> {};
TYPED_TEST_SUITE_P(GenericCSVMKernelFunction);

TYPED_TEST_P(GenericCSVMKernelFunction, predict_values) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
        params.gamma = 1.0;
        params.coef0 = 0.0;
    }

    // create the data that should be used
    const plssvm::soa_matrix<plssvm::real_type> support_vectors{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                   { plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                   { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 } },
                                                                   { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } } },
                                                                 plssvm::PADDING_SIZE,
                                                                 plssvm::PADDING_SIZE };
    const plssvm::aos_matrix<plssvm::real_type> weights{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                           { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                         plssvm::PADDING_SIZE,
                                                         plssvm::PADDING_SIZE };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } };
    plssvm::soa_matrix<plssvm::real_type> w{};
    const plssvm::soa_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 } },
                                                        { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                      plssvm::PADDING_SIZE,
                                                      plssvm::PADDING_SIZE };

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::predict_values is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // predict the values using the previously learned support vectors and weights
    const plssvm::aos_matrix<plssvm::real_type> calculated = svm.predict_values(params, support_vectors, weights, rho, w, data);

    // calculate correct predict values
    plssvm::soa_matrix<plssvm::real_type> correct_w;
    if (kernel == plssvm::kernel_function_type::linear) {
        correct_w = compare::calculate_w(weights, support_vectors);
    }
    const plssvm::aos_matrix<plssvm::real_type> correct_predict_values = compare::predict_values(params, correct_w, weights, rho, support_vectors, data);

    // check the calculated result for correctness
    ASSERT_EQ(calculated.shape(), correct_predict_values.shape());
    EXPECT_FLOATING_POINT_MATRIX_NEAR(calculated, correct_predict_values);
    // in case of the linear kernel, the w vector should have been filled
    if (kernel == plssvm::kernel_function_type::linear) {
        EXPECT_EQ(w.num_rows(), rho.size());
        EXPECT_FLOATING_POINT_MATRIX_NEAR(w, (plssvm::soa_matrix<plssvm::real_type>{ weights, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    } else {
        EXPECT_TRUE(w.empty());
    }
}
TYPED_TEST_P(GenericCSVMKernelFunction, predict_values_provided_w) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        SUCCEED() << "Test is only applicable for the linear kernel function!";
    } else {
        // create parameter struct
        plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };

        // create the data that should be used
        const plssvm::soa_matrix<plssvm::real_type> support_vectors{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                       { plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                       { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 } },
                                                                       { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } } },
                                                                     plssvm::PADDING_SIZE,
                                                                     plssvm::PADDING_SIZE };
        const plssvm::aos_matrix<plssvm::real_type> weights{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                               { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                             plssvm::PADDING_SIZE,
                                                             plssvm::PADDING_SIZE };
        const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } };
        plssvm::soa_matrix<plssvm::real_type> w{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                   { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                 plssvm::PADDING_SIZE,
                                                 plssvm::PADDING_SIZE };
        const plssvm::soa_matrix<plssvm::real_type> correct_w{ w };
        const plssvm::soa_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 } },
                                                            { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                          plssvm::PADDING_SIZE,
                                                          plssvm::PADDING_SIZE };

        // the correct result
        const plssvm::aos_matrix<plssvm::real_type> correct_predict_values{ { { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                              { plssvm::real_type{ 4.0 }, plssvm::real_type{ 4.0 } } },
                                                                            plssvm::PADDING_SIZE,
                                                                            plssvm::PADDING_SIZE };

        // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::predict_values is protected
        const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

        // predict the values using the previously learned support vectors and weights
        const plssvm::aos_matrix<plssvm::real_type> calculated = svm.predict_values(params, support_vectors, weights, rho, w, data);

        // check the calculated result for correctness
        ASSERT_EQ(calculated.shape(), correct_predict_values.shape());
        EXPECT_FLOATING_POINT_MATRIX_NEAR(calculated, correct_predict_values);
        // in case of the linear kernel, the w vector should not have changed
        EXPECT_FLOATING_POINT_MATRIX_EQ(w, correct_w);
        EXPECT_FLOATING_POINT_MATRIX_NEAR(w, (plssvm::soa_matrix<plssvm::real_type>{ weights, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    }
}

TYPED_TEST_P(GenericCSVMKernelFunction, perform_dimensional_reduction) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter struct
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(6, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::perform_dimensional_reduction is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // perform dimensional reduction
    const auto [q_red, QA_cost] = svm.perform_dimensional_reduction(params, data);

    // check values for correctness
    ASSERT_EQ(q_red.size(), data.num_rows() - 1);
    EXPECT_FLOATING_POINT_VECTOR_NEAR(q_red, compare::perform_dimensional_reduction(params, data));
    const plssvm::real_type correct_QA_cost = compare::kernel_function(params, data, data.num_rows() - 1, data, data.num_rows() - 1) + plssvm::real_type{ 1.0 } / plssvm::real_type{ params.cost };
    EXPECT_FLOATING_POINT_NEAR(QA_cost, correct_QA_cost);
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVMKernelFunction,
                            predict_values, predict_values_provided_w,
                            perform_dimensional_reduction);
// clang-format on

//*************************************************************************************************************************************//
//                                     CSVM tests depending on the solver and kernel function type                                     //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMSolverKernelFunction : public GenericCSVM<T> {};
TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunction);

TYPED_TEST_P(GenericCSVMSolverKernelFunction, solve_lssvm_system_of_linear_equations_trivial) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
        params.gamma = 1.0;
        params.coef0 = 0.0;
    } else if constexpr (kernel == plssvm::kernel_function_type::rbf) {
        GTEST_SKIP() << "solve_lssvm_system_of_linear_equations_trivial_cg_explicit currently doesn't work with the rbf kernel!";
    }

    // create the data that should be used
    // Matrix with 1-1/cost on main diagonal. Thus, the diagonal entries become one with the additional addition of 1/cost
    const plssvm::soa_matrix<plssvm::real_type> A{ { { plssvm::real_type{ std::sqrt(plssvm::real_type(1.0) - 1 / params.cost) }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                     { plssvm::real_type{ 0.0 }, plssvm::real_type{ std::sqrt(plssvm::real_type(1.0) - 1 / params.cost) }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                     { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ std::sqrt(plssvm::real_type(1.0) - 1 / params.cost) }, plssvm::real_type{ 0.0 } },
                                                     { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ std::sqrt(plssvm::real_type(1.0) - 1 / params.cost) } } },
                                                   plssvm::PADDING_SIZE,
                                                   plssvm::PADDING_SIZE };
    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                     { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } } };

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::solve_lssvm_system_of_linear_equations is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // solve the system of linear equations using the CG algorithm:
    // | Q  1 |  *  | a |  =  | y |
    // | 1  0 |     | b |     | 0 |
    // with Q = A^TA
    const auto &[calculated_x, calculated_rho, num_iter] = svm.solve_lssvm_system_of_linear_equations(A, B, params, plssvm::epsilon = 0.00001, plssvm::solver = plssvm::solver_type::cg_explicit);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(calculated_x, (plssvm::aos_matrix<plssvm::real_type>{ B, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }));
    EXPECT_TRUE(std::all_of(calculated_rho.cbegin(), calculated_rho.cend(), [front = calculated_rho.front()](const plssvm::real_type rho) { return rho == front; }));
    for (const auto rho : calculated_rho) {
        EXPECT_FLOATING_POINT_NEAR(std::abs(rho) - std::numeric_limits<plssvm::real_type>::epsilon(), std::numeric_limits<plssvm::real_type>::epsilon());
    }
    EXPECT_GT(num_iter, 0);
}
TYPED_TEST_P(GenericCSVMSolverKernelFunction, solve_lssvm_system_of_linear_equations) {
    GTEST_SKIP() << "Currently not implemented!";
}
TYPED_TEST_P(GenericCSVMSolverKernelFunction, solve_lssvm_system_of_linear_equations_with_correction) {
    GTEST_SKIP() << "Currently not implemented!";
}

TYPED_TEST_P(GenericCSVMSolverKernelFunction, assemble_kernel_matrix_minimal) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 1.0 };
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
        params.gamma = 1.0;
        params.coef0 = 0.0;
    } else if constexpr (kernel == plssvm::kernel_function_type::rbf) {
        params.gamma = 1.0;
    }
    const plssvm::soa_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                        { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                        { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } },
                                                      plssvm::PADDING_SIZE,
                                                      plssvm::PADDING_SIZE };  // 3 x 3 -> assemble 2 x 2 matrix
    [[maybe_unused]] const std::vector<plssvm::real_type> q_red(data.num_cols() - 1, plssvm::real_type{ 0.0 });
    [[maybe_unused]] const plssvm::real_type QA_cost{ 0.0 };

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::assemble_kernel_matrix is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ASSERT_ENABLED)
        EXPECT_DEATH(std::ignore = svm.setup_data_on_devices(solver, data), "An explicit solver type must be provided instead of solver_type::automatic!");
#else
        SUCCEED() << "Solver type is automatic, but assertions are disabled!";
#endif
    } else {
        // assemble the kernel matrix
        const plssvm::detail::simple_any data_d = svm.setup_data_on_devices(solver, data);
        const plssvm::detail::simple_any kernel_matrix_d = svm.assemble_kernel_matrix(solver, params, data_d, q_red, QA_cost);

        // get result based on used backend
        std::vector<plssvm::real_type> kernel_matrix{};
        if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp) {
            kernel_matrix = kernel_matrix_d.get<std::vector<plssvm::real_type>>();  // std::vector
#if defined(PLSSVM_USE_GEMM)
            ASSERT_EQ(kernel_matrix.size(), 4);
#else
            ASSERT_EQ(kernel_matrix.size(), 3);
#endif
        } else {
            const auto &kernel_matrix_d_ptr = kernel_matrix_d.get<device_ptr_type>();  // device_ptr -> convert it to a std::vector
            kernel_matrix.resize(kernel_matrix_d_ptr.size());
            kernel_matrix_d_ptr.copy_to_host(kernel_matrix);
#if defined(PLSSVM_USE_GEMM)
            ASSERT_EQ(kernel_matrix.size(), (2 + plssvm::PADDING_SIZE) * (2 + plssvm::PADDING_SIZE));
#else
            ASSERT_EQ(kernel_matrix.size(), (2 + plssvm::PADDING_SIZE) * (2 + plssvm::PADDING_SIZE + 1) / 2);
#endif
        }

        std::vector<plssvm::real_type> ground_truth{};
        if constexpr (kernel == plssvm::kernel_function_type::rbf) {
#if defined(PLSSVM_USE_GEMM)
            ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ plssvm::real_type{ 2.0 }, static_cast<plssvm::real_type>(std::exp(-27.0)), static_cast<plssvm::real_type>(std::exp(-27.0)), plssvm::real_type{ 2.0 } }, { 2, 2 });
#else
            ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ plssvm::real_type{ 2.0 }, static_cast<plssvm::real_type>(std::exp(-27.0)), plssvm::real_type{ 2.0 } }, { 2, 2 });
#endif
        } else {
#if defined(PLSSVM_USE_GEMM)
            ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ plssvm::real_type{ 15.0 }, plssvm::real_type{ 32.0 }, plssvm::real_type{ 32.0 }, plssvm::real_type{ 78.0 } }, { 2, 2 });
#else
            ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ plssvm::real_type{ 15.0 }, plssvm::real_type{ 32.0 }, plssvm::real_type{ 78.0 } }, { 2, 2 });
#endif
        }

        EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, ground_truth);
    }
}
TYPED_TEST_P(GenericCSVMSolverKernelFunction, assemble_kernel_matrix) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    plssvm::parameter params{ plssvm::kernel_type = kernel };
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.gamma = plssvm::real_type{ 1.0 / 3.0 };
        params.coef0 = 1.0;
    } else if constexpr (kernel == plssvm::kernel_function_type::rbf) {
        params.gamma = plssvm::real_type{ 1.0 / 3.0 };
    }
    const plssvm::soa_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                        { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                        { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } },
                                                      plssvm::PADDING_SIZE,
                                                      plssvm::PADDING_SIZE };
    [[maybe_unused]] const std::vector<plssvm::real_type> q_red = { plssvm::real_type{ 3.0 }, plssvm::real_type{ 4.0 } };
    [[maybe_unused]] const plssvm::real_type QA_cost{ 2.0 };

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::assemble_kernel_matrix is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ASSERT_ENABLED)
        EXPECT_DEATH(std::ignore = svm.setup_data_on_devices(solver, data), "An explicit solver type must be provided instead of solver_type::automatic!");
#else
        SUCCEED() << "Solver type is automatic, but assertions are disabled!";
#endif
    } else {
        // assemble the kernel matrix
        const plssvm::detail::simple_any data_d = svm.setup_data_on_devices(solver, data);
        const plssvm::detail::simple_any kernel_matrix_d = svm.assemble_kernel_matrix(solver, params, data_d, q_red, QA_cost);

        // get result based on used backend
        std::vector<plssvm::real_type> kernel_matrix{};
        if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp) {
            kernel_matrix = kernel_matrix_d.get<std::vector<plssvm::real_type>>();  // std::vector
#if defined(PLSSVM_USE_GEMM)
            ASSERT_EQ(kernel_matrix.size(), 4);
#else
            ASSERT_EQ(kernel_matrix.size(), 3);
#endif
        } else {
            const auto &kernel_matrix_d_ptr = kernel_matrix_d.get<device_ptr_type>();  // device_ptr -> convert it to a std::vector
            kernel_matrix.resize(kernel_matrix_d_ptr.size());
            kernel_matrix_d_ptr.copy_to_host(kernel_matrix);
#if defined(PLSSVM_USE_GEMM)
            ASSERT_EQ(kernel_matrix.size(), (2 + plssvm::PADDING_SIZE) * (2 + plssvm::PADDING_SIZE));
#else
            ASSERT_EQ(kernel_matrix.size(), (2 + plssvm::PADDING_SIZE) * (2 + plssvm::PADDING_SIZE + 1) / 2);
#endif
        }

        std::vector<plssvm::real_type> ground_truth{};
        switch (kernel) {
            case plssvm::kernel_function_type::linear:
#if defined(PLSSVM_USE_GEMM)
                ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ plssvm::real_type{ 11.0 }, plssvm::real_type{ 27.0 }, plssvm::real_type{ 27.0 }, plssvm::real_type{ 72.0 } }, { 2, 2 });
#else
                ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ plssvm::real_type{ 11.0 }, plssvm::real_type{ 27.0 }, plssvm::real_type{ 72.0 } }, { 2, 2 });
#endif
                break;
            case plssvm::kernel_function_type::polynomial:
#if defined(PLSSVM_USE_GEMM)
                ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ static_cast<plssvm::real_type>(std::pow(14.0 / 3.0 + 1.0, 3) - 3.0), static_cast<plssvm::real_type>(std::pow(32.0 / 3.0 + 1.0, 3) - 5.0), static_cast<plssvm::real_type>(std::pow(32.0 / 3.0 + 1.0, 3) - 5.0), static_cast<plssvm::real_type>(std::pow(77.0 / 3.0 + 1.0, 3) + 1.0 - 6.0) }, { 2, 2 });
#else
                ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ static_cast<plssvm::real_type>(std::pow(14.0 / 3.0 + 1.0, 3) - 3.0), static_cast<plssvm::real_type>(std::pow(32.0 / 3.0 + 1.0, 3) - 5.0), static_cast<plssvm::real_type>(std::pow(77.0 / 3.0 + 1.0, 3) + 1.0 - 6.0) }, { 2, 2 });

#endif
                break;
            case plssvm::kernel_function_type::rbf:
#if defined(PLSSVM_USE_GEMM)
                ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ plssvm::real_type{ -2.0 }, static_cast<plssvm::real_type>(std::exp(-9.0) - 5.0), static_cast<plssvm::real_type>(std::exp(-9.0) - 5.0), plssvm::real_type{ -4.0 } }, { 2, 2 });
#else
                ground_truth = util::pad_1D_vector<csvm_type>(std::vector<plssvm::real_type>{ plssvm::real_type{ -2.0 }, static_cast<plssvm::real_type>(std::exp(-9.0) - 5.0), plssvm::real_type{ -4.0 } }, { 2, 2 });
#endif
                break;
        }
        EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, ground_truth);
    }
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunction,
                            solve_lssvm_system_of_linear_equations_trivial, solve_lssvm_system_of_linear_equations, solve_lssvm_system_of_linear_equations_with_correction,
                            assemble_kernel_matrix_minimal, assemble_kernel_matrix);
// clang-format on

//*************************************************************************************************************************************//
//                                 CSVM tests depending on the kernel function and classification type                                 //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMKernelFunctionClassification : public GenericCSVM<T> {};
TYPED_TEST_SUITE_P(GenericCSVMKernelFunctionClassification);

TYPED_TEST_P(GenericCSVMKernelFunctionClassification, predict) {
    // TODO: change from hardcoded file to logic test

    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::classification_type classification = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter struct
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create data set to be used
    const plssvm::data_set<label_type> test_data{ PLSSVM_TEST_PATH "/data/predict/50x20.libsvm" };

    // read the previously learned model
    const plssvm::model<label_type> model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/50x20_{}_{}_{}.libsvm.model", plssvm::detail::arithmetic_type_name<plssvm::real_type>(), kernel, classification) };

    // create C-SVM
    const csvm_type svm = util::construct_from_tuple<csvm_type>(params, csvm_test_type::additional_arguments);

    // predict label
    const std::vector<label_type> calculated = svm.predict(model, test_data);

    // read ground truth from file
    plssvm::detail::io::file_reader reader{ PLSSVM_TEST_PATH "/data/predict/50x20.libsvm.predict" };
    reader.read_lines();
    std::vector<label_type> ground_truth(reader.num_lines());
    for (std::size_t i = 0; i < reader.num_lines(); ++i) {
        ground_truth[i] = plssvm::detail::convert_to<label_type>(reader.line(i));
    }

    // check the calculated result for correctness
    EXPECT_EQ(calculated, ground_truth);
}
TYPED_TEST_P(GenericCSVMKernelFunctionClassification, score_model) {
    // TODO: change from hardcoded file to logic test

    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::classification_type classification = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };

    // read the previously learned model
    const plssvm::model<label_type> model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/50x20_{}_{}_{}.libsvm.model", plssvm::detail::arithmetic_type_name<plssvm::real_type>(), kernel, classification) };

    // create C-SVM
    const csvm_type svm = util::construct_from_tuple<csvm_type>(params, csvm_test_type::additional_arguments);

    // score model
    const plssvm::real_type calculated = svm.score(model);

    // check the calculated result for correctness
    EXPECT_EQ(calculated, plssvm::real_type{ 1.0 });
}
TYPED_TEST_P(GenericCSVMKernelFunctionClassification, score) {
    // TODO: change from hardcoded file to logic test

    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::classification_type classification = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create data set to be used
    const plssvm::data_set<label_type> test_data{ PLSSVM_TEST_PATH "/data/predict/50x20.libsvm" };

    // read the previously learned model
    const plssvm::model<label_type> model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/50x20_{}_{}_{}.libsvm.model", plssvm::detail::arithmetic_type_name<plssvm::real_type>(), kernel, classification) };

    // create C-SVM
    const csvm_type svm = util::construct_from_tuple<csvm_type>(params, csvm_test_type::additional_arguments);

    // score the test data using the learned model
    const plssvm::real_type calculated = svm.score(model, test_data);

    // check the calculated result for correctness
    // it doesn't converge for float, linear, OAO
    EXPECT_EQ(calculated, plssvm::real_type{ 1.0 });
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVMKernelFunctionClassification,
                            predict, score_model, score);
// clang-format on

//*************************************************************************************************************************************//
//                             CSVM tests depending on the solver, kernel function, and classification type                            //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMSolverKernelFunctionClassification : public GenericCSVM<T> {};
TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunctionClassification);

TYPED_TEST_P(GenericCSVMSolverKernelFunctionClassification, fit) {
    // TODO: change from hardcoded file to logic test

    using label_type = util::test_parameter_type_at_t<1, TypeParam>;
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;
    constexpr plssvm::classification_type classification = util::test_parameter_value_at_v<2, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    // create parameter struct
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create data set to be used
    const plssvm::data_set<label_type> test_data{ PLSSVM_TEST_PATH "/data/predict/50x20.libsvm" };

    // read the previously learned model
    const plssvm::model<label_type> correct_model{ fmt::format(PLSSVM_TEST_PATH "/data/predict/50x20_{}_{}_{}.libsvm.model", plssvm::detail::arithmetic_type_name<plssvm::real_type>(), kernel, classification) };

    // create C-SVM
    const csvm_type svm = util::construct_from_tuple<csvm_type>(params, csvm_test_type::additional_arguments);

    // call fit
    const plssvm::model<label_type> model = svm.fit(test_data, plssvm::epsilon = 1e-10, plssvm::solver = solver, plssvm::classification = classification);

    // check the calculated result for correctness
    ASSERT_EQ(model.num_support_vectors(), correct_model.num_support_vectors());
    ASSERT_EQ(model.num_features(), correct_model.num_features());
    ASSERT_EQ(model.num_classes(), correct_model.num_classes());
    // can't check support vectors for equality since the SV order after our IO is non-deterministic
    ASSERT_EQ(model.weights().size(), correct_model.weights().size());
    // can't check weights for equality since the SV order after our IO is non-deterministic
    // TODO: the eps factor must be selected WAY too large
    //    EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(model.rho(), correct_model.rho(), plssvm::real_type{ 1e12 });
    EXPECT_EQ(model.get_classification_type(), classification);
    EXPECT_TRUE(model.num_iters().has_value());
    EXPECT_EQ(model.num_iters().value().size(), (plssvm::calculate_number_of_classifiers(classification, correct_model.num_classes())));
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunctionClassification,
                            fit);
// clang-format on

//*************************************************************************************************************************************//
//                                                           CSVM DeathTests                                                           //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMSolverDeathTest : public GenericCSVMSolver<T> {};
TYPED_TEST_SUITE_P(GenericCSVMSolverDeathTest);

TYPED_TEST_P(GenericCSVMSolverDeathTest, conjugate_gradients_empty_B) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm_type csvm{};

    const std::array<std::size_t, 2> shape{ 4, 4 };
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> matr_A(shape[0] * shape[1]);
#else
    const std::vector<plssvm::real_type> matr_A(shape[0] * (shape[1] + 1) / 2);
#endif
    const plssvm::detail::simple_any A{ util::init_matrix<csvm_type, device_ptr_type>(matr_A, csvm, shape) };

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> empty_matr{};

    EXPECT_DEATH(std::ignore = csvm.conjugate_gradients(A, empty_matr, plssvm::real_type{ 0.001 }, 6, solver), "The right-hand sides may not be empty!");
}
TYPED_TEST_P(GenericCSVMSolverDeathTest, conjugate_gradients_invalid_eps) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type csvm{};

    const std::array<std::size_t, 2> shape{ 4, 4 };
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> matr_A(shape[0] * shape[1]);
#else
    const std::vector<plssvm::real_type> matr_A(shape[0] * (shape[1] + 1) / 2);
#endif
    const plssvm::detail::simple_any A{ util::init_matrix<csvm_type, device_ptr_type>(matr_A, csvm, shape) };
    const plssvm::soa_matrix<plssvm::real_type> B{ 1, 6 };

    EXPECT_DEATH(std::ignore = csvm.conjugate_gradients(A, B, 0.0, 6, solver), "The epsilon value must be greater than 0.0!");
    EXPECT_DEATH(std::ignore = csvm.conjugate_gradients(A, B, -0.5, 6, solver), "The epsilon value must be greater than 0.0!");
}
TYPED_TEST_P(GenericCSVMSolverDeathTest, conjugate_gradients_invalid_max_cg_iter) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type csvm{};

    const std::array<std::size_t, 2> shape{ 4, 4 };
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> matr_A(shape[0] * shape[1]);
#else
    const std::vector<plssvm::real_type> matr_A(shape[0] * (shape[1] + 1) / 2);
#endif
    const plssvm::detail::simple_any A{ util::init_matrix<csvm_type, device_ptr_type>(matr_A, csvm, shape) };
    const plssvm::soa_matrix<plssvm::real_type> B{ 1, 6 };

    EXPECT_DEATH(std::ignore = csvm.conjugate_gradients(A, B, plssvm::real_type{ 0.001 }, 0, solver), "The maximum number of iterations must be greater than 0!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, run_blas_level_3_empty_B) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type csvm{};

    const std::array<std::size_t, 2> shape{ 4, 4 };
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> matr_A(shape[0] * shape[1]);
#else
    const std::vector<plssvm::real_type> matr_A(shape[0] * (shape[1] + 1) / 2);
#endif
    const plssvm::detail::simple_any A{ util::init_matrix<csvm_type, device_ptr_type>(matr_A, csvm, shape) };
    const plssvm::soa_matrix<plssvm::real_type> empty_matr{};
    plssvm::soa_matrix<plssvm::real_type> C{ 4, 4 };

    EXPECT_DEATH(std::ignore = csvm.run_blas_level_3(solver, plssvm::real_type{ 1.0 }, A, empty_matr, plssvm::real_type{ 1.0 }, C), "The B matrix must not be empty!");
}
TYPED_TEST_P(GenericCSVMSolverDeathTest, run_blas_level_3_empty_C) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type csvm{};

    const std::array<std::size_t, 2> shape{ 4, 4 };
#if defined(PLSSVM_USE_GEMM)
    const std::vector<plssvm::real_type> matr_A(shape[0] * shape[1]);
#else
    const std::vector<plssvm::real_type> matr_A(shape[0] * (shape[1] + 1) / 2);
#endif
    const plssvm::detail::simple_any A{ util::init_matrix<csvm_type, device_ptr_type>(matr_A, csvm, shape) };
    const plssvm::soa_matrix<plssvm::real_type> B{ 4, 4 };
    plssvm::soa_matrix<plssvm::real_type> empty_matr{};

    EXPECT_DEATH(std::ignore = csvm.run_blas_level_3(solver, plssvm::real_type{ 1.0 }, A, B, plssvm::real_type{ 1.0 }, empty_matr), "The C matrix must not be empty!");
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolverDeathTest,
                            conjugate_gradients_empty_B, conjugate_gradients_invalid_eps, conjugate_gradients_invalid_max_cg_iter,
                            run_blas_level_3_empty_B, run_blas_level_3_empty_C);
// clang-format on

template <typename T>
class GenericCSVMKernelFunctionDeathTest : public GenericCSVMKernelFunction<T> {};
TYPED_TEST_SUITE_P(GenericCSVMKernelFunctionDeathTest);

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_empty_A) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> empty_matr{};
    const plssvm::aos_matrix<plssvm::real_type> B{ 1, 4 };

    EXPECT_DEATH(std::ignore = csvm.solve_lssvm_system_of_linear_equations(empty_matr, B, params), "The A matrix may not be empty!");
}
TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_A_without_padding) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> A{ 6, 4 };
    const plssvm::aos_matrix<plssvm::real_type> B{ 1, 4 };

    EXPECT_DEATH(std::ignore = csvm.solve_lssvm_system_of_linear_equations(A, B, params), "The A matrix must be padded!");
}
TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_A_wrong_padding_sizes) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> A{ 6, 4, 0, 1 };
    const plssvm::aos_matrix<plssvm::real_type> B{ 1, 4 };

    EXPECT_DEATH(std::ignore = csvm.solve_lssvm_system_of_linear_equations(A, B, params),
                 ::testing::HasSubstr(fmt::format("The provided matrix must be padded with [{}, {}], but is padded with [0, 1]!", plssvm::PADDING_SIZE, plssvm::PADDING_SIZE)));
}
TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_empty_B) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_matr{};
    const plssvm::soa_matrix<plssvm::real_type> A{ 6, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE };

    EXPECT_DEATH(std::ignore = csvm.solve_lssvm_system_of_linear_equations(A, empty_matr, params), "The B matrix may not be empty!");
}
TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_size_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> A{ 6, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE };
    const plssvm::aos_matrix<plssvm::real_type> B{ 1, 3 };

    EXPECT_DEATH(std::ignore = csvm.solve_lssvm_system_of_linear_equations(A, B, params), ::testing::HasSubstr("The number of data points in A (6) and B (3) must be the same!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, perform_dimensional_reduction_empty_A) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> empty_matr{};

    EXPECT_DEATH(std::ignore = csvm.perform_dimensional_reduction(params, empty_matr), "The matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_empty_matrices) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_aos_matr{};
    const plssvm::soa_matrix<plssvm::real_type> empty_soa_matr{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const std::vector<plssvm::real_type> rho(2);
    plssvm::soa_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // support vectors shouldn't be empty
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, empty_soa_matr, weights, rho, w, data), "The support vectors must not be empty!");
    // weights shouldn't be empty
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, empty_aos_matr, rho, w, data), ::testing::HasSubstr("The alpha vectors (weights) must not be empty!"));
    // predict points shouldn't be empty
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, weights, rho, w, empty_soa_matr), "The data points to predict must not be empty!");
}
TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_missing_padding) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_aos_matr{};
    const plssvm::soa_matrix<plssvm::real_type> empty_soa_matr{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const auto support_vectors_without_padding = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 4);
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const auto weights_without_padding = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 4);
    const std::vector<plssvm::real_type> rho(2);
    auto w = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    auto w_without_padding = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 4);
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const auto data_without_padding = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 4);

    // support vectors must be padded
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors_without_padding, weights, rho, w, data), "The support vectors must be padded!");
    // weights must be padded
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, weights_without_padding, rho, w, data), ::testing::HasSubstr("The alpha vectors (weights) must be padded!"));
    // w must be padded
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, weights, rho, w_without_padding, data_without_padding), "Either w must be empty or must be padded!");
    // predict points must be padded
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, weights, rho, w, data_without_padding), "The data points to predict must be padded!");
}
TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_sv_alpha_size_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(3, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const std::vector<plssvm::real_type> rho(2);
    plssvm::soa_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // the number of support vectors and weights must be identical
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("The number of support vectors (3) and number of weights (4) must be the same!"));
}
TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_rho_alpha_size_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const std::vector<plssvm::real_type> rho(1);
    plssvm::soa_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // the number of rho values and weight vectors must be identical
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("The number of rho values (1) and the number of weight vectors (2) must be the same!"));
}
TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_w_size_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const std::vector<plssvm::real_type> rho(2);
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // the number of features and w values must be identical
    auto w = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 3, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("Either w must be empty or contain exactly the same number of values (3) as features are present (4)!"));
    // the number of weight and w vectors must be identical
    w = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(3, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("Either w must be empty or contain exactly the same number of vectors (3) as the alpha vector (2)!"));
}
TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_num_features_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type csvm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(4, 5, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);
    const std::vector<plssvm::real_type> rho(2);
    plssvm::soa_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(2, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE);

    // the number of features for the support vectors and predict points must be identical
    EXPECT_DEATH(std::ignore = csvm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("The number of features in the support vectors (5) must be the same as in the data points to predict (4)!"));
}

// clang-format off
REGISTER_TYPED_TEST_SUITE_P(GenericCSVMKernelFunctionDeathTest,
                            solve_lssvm_system_of_linear_equations_empty_A, solve_lssvm_system_of_linear_equations_A_without_padding, solve_lssvm_system_of_linear_equations_A_wrong_padding_sizes, solve_lssvm_system_of_linear_equations_empty_B, solve_lssvm_system_of_linear_equations_size_mismatch,
                            perform_dimensional_reduction_empty_A,
                            predict_values_empty_matrices, predict_values_missing_padding, predict_values_sv_alpha_size_mismatch, predict_values_rho_alpha_size_mismatch, predict_values_w_size_mismatch, predict_values_num_features_mismatch);
// clang-format on

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_CSVM_TESTS_HPP_