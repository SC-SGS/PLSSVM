/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Generic C-SVM tests for all backends to reduce code duplication.
 */

#ifndef PLSSVM_TESTS_BACKENDS_GENERIC_BASE_CSVM_TESTS_HPP_
#define PLSSVM_TESTS_BACKENDS_GENERIC_BASE_CSVM_TESTS_HPP_
#pragma once

#include "plssvm/backend_types.hpp"                     // plssvm::backend_type
#include "plssvm/backends/execution_range.hpp"          // plssvm::detail::execution_range
#include "plssvm/backends/Kokkos/execution_spaces.hpp"  // plssvm::kokkos::execution_space
#include "plssvm/constants.hpp"                         // plssvm::real_type
#include "plssvm/detail/data_distribution.hpp"          // plssvm::detail::{triangular_data_distribution, rectangular_data_distribution}
#include "plssvm/detail/memory_size.hpp"                // memory size literals
#include "plssvm/detail/move_only_any.hpp"              // plssvm::detail::move_only_any
#include "plssvm/detail/utility.hpp"                    // plssvm::detail::{unreachable, get}
#include "plssvm/kernel_function_types.hpp"             // plssvm::csvm_to_backend_type_v, plssvm::backend_type
#include "plssvm/matrix.hpp"                            // plssvm::aos_matrix
#include "plssvm/mpi/communicator.hpp"                  // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                         // plssvm::parameter
#include "plssvm/shape.hpp"                             // plssvm::shape
#include "plssvm/solver_types.hpp"                      // plssvm::solver_type
#include "plssvm/target_platforms.hpp"                  // plssvm::target_platform

#include "tests/backends/ground_truth.hpp"  // ground_truth::{kernel_function, perform_dimensional_reduction}
#include "tests/custom_test_macros.hpp"     // EXPECT_FLOATING_POINT_MATRIX_EQ, EXPECT_FLOATING_POINT_VECTOR_NEAR, EXPECT_FLOATING_POINT_NEAR
#include "tests/types_to_test.hpp"          // util::{test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"                // util::{redirect_output, generate_specific_matrix, construct_from_tuple, generate_random_matrix}

#include "fmt/format.h"   // fmt::format
#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_EQ, EXPECT_NE, EXPECT_GT, EXPECT_TRUE, EXPECT_DEATH,
                          // ASSERT_EQ, SUCCEED, ::testing::Test

#include <cmath>       // std::sqrt, std::abs
#include <cstddef>     // std::size_t
#include <cstring>     // std::memcpy
#include <functional>  // std::cref
#include <limits>      // std::numeric_limits::epsilon
#include <memory>      // std::unique_ptr, std::make_unique
#include <optional>    // std::optional
#include <tuple>       // std::ignore, std::tuple, std::make_tuple
#include <utility>     // std::move
#include <vector>      // std::vector

namespace util {

/**
 * @brief Initialize the explicit kernel matrices using the values provided by @p matr based on the @p csvm_type.
 * @tparam csvm_type the PLSSVM C-SVM backend type
 * @tparam device_ptr_type the device pointer type; only used if a GPU backend is used
 * @tparam matrix_type the type of the matrix
 * @tparam used_csvm_type the type of the @p csvm; may also be a Mock C-SVM!
 * @tparam Args the types of the additional arguments; none for the explicit matrix
 * @param[in] matr the matrix to create the explicit kernel matrix from
 * @param[in] csvm the C-SVM encapsulating the device on which the matrix should be allocated
 * @return a std::vector of `plssvm::detail::move_only_any` with a wrapped value usable in the PLSSVM functions (`[[nodiscard]]`)
 */
template <typename csvm_type, typename device_ptr_type, typename matrix_type, typename used_csvm_type, typename... Args>
[[nodiscard]] inline std::vector<plssvm::detail::move_only_any> init_explicit_matrices(const matrix_type &matr, used_csvm_type &csvm) {
    using real_type = typename matrix_type::value_type;
    std::vector<plssvm::detail::move_only_any> result(csvm.num_available_devices());

    // function to split the provided matrix into a device specific partial matrix
    const auto calculate_partial_kernel_matrix = [&matr](const std::size_t row_offset, const std::size_t device_specific_num_rows) {
        std::vector<real_type> partial_kernel_matrix;
        for (std::size_t row = row_offset; row < row_offset + device_specific_num_rows; ++row) {
            for (std::size_t col = row; col < matr.num_cols(); ++col) {
                partial_kernel_matrix.push_back(matr(row, col));
            }
        }

        return partial_kernel_matrix;
    };

    if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::stdpar || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::hpx) {
        // only a single device for OpenMP, stdpar, and HPX on the CPU
        const std::vector<real_type> partial_kernel_matrix = calculate_partial_kernel_matrix(0, matr.num_rows());
        auto ptr = std::make_unique<real_type[]>(partial_kernel_matrix.size());  // NOLINT: C-style array must be used here to align with real implementation
        std::memcpy(ptr.get(), partial_kernel_matrix.data(), partial_kernel_matrix.size() * sizeof(real_type));
        result[0] = plssvm::detail::move_only_any{ std::move(ptr) };
    } else {
        for (std::size_t device_id = 0; device_id < csvm.num_available_devices(); ++device_id) {
            auto &device = csvm.devices_[device_id];
            const std::size_t device_specific_num_rows = csvm.data_distribution_->place_specific_num_rows(device_id);
            const std::size_t row_offset = csvm.data_distribution_->place_row_offset(device_id);

            // split kernel matrix on the cpu
            const std::vector<real_type> partial_kernel_matrix = calculate_partial_kernel_matrix(row_offset, device_specific_num_rows);

            // move kernel matrix to the specific device
            device_ptr_type ptr{ partial_kernel_matrix.size(), device };
            ptr.copy_to_device(partial_kernel_matrix);
            result[device_id] = plssvm::detail::move_only_any{ std::move(ptr) };
        }
    }
    return result;
}

/**
 * @brief Initialize the implicit kernel matrices using the values provided by @p matr based on the @p csvm_type.
 * @tparam csvm_type the PLSSVM C-SVM backend type
 * @tparam device_ptr_type the device pointer type; only used if a GPU backend is used
 * @tparam matrix_type the type of the matrix
 * @tparam used_csvm_type the type of the @p csvm; may also be a Mock C-SVM!
 * @tparam Args the types of the additional arguments
 * @param[in] matr the matrix to create the explicit kernel matrix from
 * @param[in] csvm the C-SVM encapsulating the device on which the matrix should be allocated
 * @param[in] args the additional arguments
 * @return a std::vector of `plssvm::detail::move_only_any` with a wrapped value usable in the PLSSVM functions (`[[nodiscard]]`)
 */
template <typename csvm_type, typename device_ptr_type, typename matrix_type, typename used_csvm_type, typename real_type = typename matrix_type::value_type>
[[nodiscard]] inline std::vector<plssvm::detail::move_only_any> init_implicit_matrices(const matrix_type &matr, used_csvm_type &csvm, const plssvm::parameter params, const std::vector<real_type> &q_red, const real_type QA_cost) {
    std::vector<plssvm::detail::move_only_any> result(csvm.num_available_devices());

    for (std::size_t device_id = 0; device_id < csvm.num_available_devices(); ++device_id) {
        // created matrix is different for the OpenMP, stdpar or HPX backend and the GP92U backends!
        if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::stdpar || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::hpx) {
            // only a single device ever in use
            result[0] = plssvm::detail::move_only_any{ std::make_tuple(std::cref(matr), params, std::cref(q_red), QA_cost) };
        } else {
            auto &device = csvm.devices_[device_id];

            // create device pointer
            device_ptr_type matr_d{ matr.shape(), device };
            matr_d.copy_to_device(matr);

            // create device pointer for the q vector
            device_ptr_type q_d{ q_red.size(), device };
            q_d.copy_to_device(q_red, 0, q_red.size());

            // kernel launch specific sizes
            const unsigned long long device_specific_num_rows = csvm.data_distribution_->place_specific_num_rows(device_id);
            const unsigned long long device_row_offset = csvm.data_distribution_->place_row_offset(device_id);

            // the block dimension is THREAD_BLOCK_SIZE x THREAD_BLOCK_SIZE
            const plssvm::detail::dim_type block{ std::size_t{ plssvm::THREAD_BLOCK_SIZE }, std::size_t{ plssvm::THREAD_BLOCK_SIZE } };

            // define the full execution grid
            const plssvm::detail::dim_type grid{
                static_cast<std::size_t>(std::ceil(static_cast<double>(matr.shape().x - 1 - device_row_offset) / static_cast<double>(block.x))),
                static_cast<std::size_t>(std::ceil(static_cast<double>(device_specific_num_rows) / static_cast<double>(block.y)))
            };

            // create the final execution range
            const plssvm::detail::execution_range exec{ block, csvm.get_max_work_group_size(device_id), grid, csvm.get_max_grid_size(device_id) };

            result[device_id] = plssvm::detail::move_only_any{ std::make_tuple(exec, std::move(matr_d), params, std::move(q_d), QA_cost) };
        }
    }
    return result;
}

/**
 * @brief Initialize the kernel matrices based on the used @p solver type using the values provided by @p matr based on the @p csvm_type.
 * @tparam csvm_type the PLSSVM C-SVM backend type
 * @tparam device_ptr_type the device pointer type; only used if a GPU backend is used
 * @tparam matrix_type the type of the matrix
 * @tparam used_csvm_type the type of the @p csvm; may also be a Mock C-SVM!
 * @tparam Args the types of the additional arguments
 * @param[in] matr the matrix to create the explicit kernel matrix from
 * @param[in] solver the used solver type in the CG algorithm
 * @param[in] csvm the C-SVM encapsulating the device on which the matrix should be allocated
 * @param[in] args the additional arguments
 * @return a std::vector of `plssvm::detail::move_only_any` with a wrapped value usable in the PLSSVM functions (`[[nodiscard]]`)
 */
template <typename csvm_type, typename device_ptr_type, typename matrix_type, typename used_csvm_type, typename... Args>
[[nodiscard]] inline std::vector<plssvm::detail::move_only_any> init_matrices(const matrix_type &matr, const plssvm::solver_type solver, used_csvm_type &csvm, Args &&...args) {
    switch (solver) {
        case plssvm::solver_type::automatic:
            {
                std::vector<plssvm::detail::move_only_any> result;
                result.reserve(csvm.num_available_devices());
                for (std::size_t device_id = 0; device_id < csvm.num_available_devices(); ++device_id) {
                    result.push_back(plssvm::detail::move_only_any{ std::vector<typename matrix_type::value_type>{} });
                }
                return result;  // dummy return only necessary for the DeathTests -> VALUE NOT USED!
            }
        case plssvm::solver_type::cg_explicit:
            // no additional arguments are used
            return init_explicit_matrices<csvm_type, device_ptr_type>(matr, csvm);
        case plssvm::solver_type::cg_implicit:
            // additional arguments are: params, q_red, QA_cost
            return init_implicit_matrices<csvm_type, device_ptr_type>(matr, csvm, std::forward<Args>(args)...);
    }
    // should never be reached!
    plssvm::detail::unreachable();
}

}  // namespace util

//*************************************************************************************************************************************//
//                                                  C-SVM tests depending on nothing                                                   //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVM : public ::testing::Test,
                    protected util::redirect_output<> { };

TYPED_TEST_SUITE_P(GenericCSVM);

TYPED_TEST_P(GenericCSVM, GetTargetPlatform) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create normal C-SVM
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // after construction: get_target_platform must refer to a plssvm::target_platform that is not automatic
    EXPECT_NE(svm.get_target_platform(), plssvm::target_platform::automatic);
}

TYPED_TEST_P(GenericCSVM, NumAvailableDevices) {
    using namespace plssvm::detail::literals;  // NOLINT(google-build-using-namespace): only imports custom user-defined literals into this namespace
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create normal C-SVM
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the maximum memory allocation size should be greater than 0!
    if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::stdpar || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::hpx) {
        EXPECT_EQ(svm.num_available_devices(), 1);
    } else {
        EXPECT_GE(svm.num_available_devices(), 1);
    }
}

TYPED_TEST_P(GenericCSVM, GetDeviceMemory) {
    using namespace plssvm::detail::literals;  // NOLINT(google-build-using-namespace): only imports custom user-defined literals into this namespace
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the available device memory should be greater than 0!
    const std::vector<plssvm::detail::memory_size> mem = svm.get_device_memory();
    EXPECT_GE(mem.size(), 1);
    for (const plssvm::detail::memory_size ms : mem) {
        EXPECT_GT(ms, 0_B);
    }
}

TYPED_TEST_P(GenericCSVM, GetMaxMemAllocSize) {
    using namespace plssvm::detail::literals;  // NOLINT(google-build-using-namespace): only imports custom user-defined literals into this namespace
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the maximum memory allocation size should be greater than 0!
    const std::vector<plssvm::detail::memory_size> mem = svm.get_max_mem_alloc_size();
    EXPECT_GE(mem.size(), 1);
    for (const plssvm::detail::memory_size ms : mem) {
        EXPECT_GT(ms, 0_B);
    }
}

TYPED_TEST_P(GenericCSVM, GetLocalMemory) {
    using namespace plssvm::detail::literals;  // NOLINT(google-build-using-namespace): only imports custom user-defined literals into this namespace
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the maximum memory allocation size should be greater than 0!
    const std::vector<std::optional<plssvm::detail::memory_size>> local_mem = svm.get_local_memory();
    EXPECT_GE(local_mem.size(), 1);
    for (const std::optional<plssvm::detail::memory_size> ms : local_mem) {
        if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::stdpar || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::hpx) {
            // some backends inherently do not have the capability of local memory
            EXPECT_EQ(ms, std::nullopt);
        } else if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::kokkos) {
            // for the Kokkos backend, the existence of the local memory size depends on the kokkos execution space
            switch (svm.get_execution_space()) {
                case plssvm::kokkos::execution_space::cuda:
                case plssvm::kokkos::execution_space::hip:
                case plssvm::kokkos::execution_space::sycl:
                    EXPECT_GT(ms.value(), 0_B);
                    break;
                case plssvm::kokkos::execution_space::hpx:
                case plssvm::kokkos::execution_space::openmp:
                case plssvm::kokkos::execution_space::threads:
                case plssvm::kokkos::execution_space::serial:
                    EXPECT_EQ(ms, std::nullopt);
                    break;
                case plssvm::kokkos::execution_space::openmp_target:
                case plssvm::kokkos::execution_space::openacc:
                case plssvm::kokkos::execution_space::automatic:
                    // should currently be unreachable!
                    break;
            }
        } else {
            EXPECT_GT(ms.value(), 0_B);
        }
    }
}

TYPED_TEST_P(GenericCSVM, BlasLevel3ExplicitWithoutC) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_explicit;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    const plssvm::real_type alpha{ 1.0 };
    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 } },
          { plssvm::real_type{ 0.2 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 } },
          { plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm) };

    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } } };

    const plssvm::real_type beta{ 0.0 };
    plssvm::aos_matrix<plssvm::real_type> C{ plssvm::shape{ 3, 3 } };
    plssvm::aos_matrix<plssvm::real_type> C2{ C };

    // perform BLAS calculation
    svm.blas_level_3(solver, alpha, A, B, beta, C);

    // check C for correctness
    const plssvm::aos_matrix<plssvm::real_type> correct_C{ { { plssvm::real_type{ 1.4 }, plssvm::real_type{ 6.5 }, plssvm::real_type{ 9.8 } },
                                                             { plssvm::real_type{ 3.2 }, plssvm::real_type{ 14.6 }, plssvm::real_type{ 21.5 } },
                                                             { plssvm::real_type{ 5.0 }, plssvm::real_type{ 22.7 }, plssvm::real_type{ 33.2 } } } };
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);

    // check wrapper function
    std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);
    EXPECT_FLOATING_POINT_MATRIX_EQ(C2, C);
}

TYPED_TEST_P(GenericCSVM, BlasLevel3Explicit) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_explicit;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    const plssvm::real_type alpha{ 1.0 };

    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 } },
          { plssvm::real_type{ 0.2 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 } },
          { plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm) };

    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } } };

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, 3 });
    plssvm::aos_matrix<plssvm::real_type> C2{ C };

    // perform BLAS calculation
    svm.blas_level_3(solver, alpha, A, B, beta, C);
    std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);

    // check C for correctness
    const plssvm::aos_matrix<plssvm::real_type> correct_C{ { { plssvm::real_type{ 1.45 }, plssvm::real_type{ 6.6 }, plssvm::real_type{ 9.95 } },
                                                             { plssvm::real_type{ 3.75 }, plssvm::real_type{ 15.2 }, plssvm::real_type{ 22.15 } },
                                                             { plssvm::real_type{ 6.05 }, plssvm::real_type{ 23.8 }, plssvm::real_type{ 34.35 } } } };

    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);
    EXPECT_FLOATING_POINT_MATRIX_EQ(C2, C);
}

TYPED_TEST_P(GenericCSVM, ConjugateGradientsTrivial) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    // solver type doesn't mather since we want to test the surrounding CG algorithm here
    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_explicit;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // create the data that should be used
    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
          { plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
          { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 } },
          { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm) };

    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                     { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } } };

    // solve AX = B
    const auto [X, num_iters] = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.00001 }, 4, solver);

    // check result
    EXPECT_FLOATING_POINT_MATRIX_NEAR(X, B);
    EXPECT_THAT(num_iters, ::testing::Each(::testing::Gt(0)));
}

TYPED_TEST_P(GenericCSVM, ConjugateGradients) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    // solver type doesn't mather since we want to test the surrounding CG algorithm here
    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_explicit;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // create the data that should be used
    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 4.0 }, plssvm::real_type{ 1.0 } },
          { plssvm::real_type{ 1.0 }, plssvm::real_type{ 3.0 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A = util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm);

    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 } },
                                                     { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 } } } };
    const plssvm::aos_matrix<plssvm::real_type> correct_X{ { { plssvm::real_type{ 1.0 / 11.0 }, plssvm::real_type{ 7.0 / 11.0 } },
                                                             { plssvm::real_type{ 1.0 / 11.0 }, plssvm::real_type{ 7.0 / 11.0 } } } };

    // solve AX = B
    const auto [X, num_iters] = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.00001 }, 2, solver);

    // check result
    EXPECT_FLOATING_POINT_MATRIX_NEAR(X, correct_X);
    EXPECT_THAT(num_iters, ::testing::Each(::testing::Gt(0)));
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVM,
                            GetTargetPlatform,
                            GetDeviceMemory,
                            GetLocalMemory,
                            GetMaxMemAllocSize,
                            NumAvailableDevices,
                            BlasLevel3ExplicitWithoutC,
                            BlasLevel3Explicit,
                            ConjugateGradientsTrivial,
                            ConjugateGradients);

//*************************************************************************************************************************************//
//                                          C-SVM tests depending on the kernel function type                                          //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMKernelFunction : public GenericCSVM<T> { };

TYPED_TEST_SUITE_P(GenericCSVMKernelFunction);

TYPED_TEST_P(GenericCSVMKernelFunction, BlasLevel3AssemblyImplicitWithoutC) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // set solver type explicitly to cg_implicit
    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_implicit;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
    }
    if constexpr (kernel == plssvm::kernel_function_type::polynomial || kernel == plssvm::kernel_function_type::sigmoid) {
        params.coef0 = 0.0;
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    const plssvm::real_type alpha{ 1.0 };

    const auto matr_A = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const auto [q, QA_cost] = ground_truth::perform_dimensional_reduction(params, matr_A);

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows() - 1, svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_implicit_matrices<csvm_type, device_ptr_type>(matr_A, svm, params, q, QA_cost) };

    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } } };

    const plssvm::real_type beta{ 0.0 };
    plssvm::aos_matrix<plssvm::real_type> C{ B.shape(), plssvm::real_type{ 0.0 } };
    plssvm::aos_matrix<plssvm::real_type> C2{ C };

    // perform BLAS calculation
    svm.blas_level_3(solver, alpha, A, B, beta, C);
    std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);

    // calculate correct results
    const plssvm::aos_matrix<plssvm::real_type> full_kernel_matrix = ground_truth::assemble_full_kernel_matrix(params, matr_A, q, QA_cost);
    plssvm::aos_matrix<plssvm::real_type> correct_C{ C.shape(), plssvm::real_type{ 0.0 } };
    ground_truth::gemm(alpha, full_kernel_matrix, B, beta, correct_C);

    // check for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);
    EXPECT_FLOATING_POINT_MATRIX_EQ(C, C2);
}

TYPED_TEST_P(GenericCSVMKernelFunction, BlasLevel3AssemblyImplicit) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // set solver type explicitly to cg_implicit
    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_implicit;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
    }
    if constexpr (kernel == plssvm::kernel_function_type::polynomial || kernel == plssvm::kernel_function_type::sigmoid) {
        params.coef0 = 0.0;
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    const plssvm::real_type alpha{ 1.0 };

    const auto matr_A = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const auto [q, QA_cost] = ground_truth::perform_dimensional_reduction(params, matr_A);

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows() - 1, svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_implicit_matrices<csvm_type, device_ptr_type>(matr_A, svm, params, q, QA_cost) };

    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } } };

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, 3 });
    plssvm::aos_matrix<plssvm::real_type> C2{ C };
    plssvm::aos_matrix<plssvm::real_type> correct_C{ C };

    // perform BLAS calculation
    svm.blas_level_3(solver, alpha, A, B, beta, C);
    std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);

    // calculate correct results
    const plssvm::aos_matrix<plssvm::real_type> kernel_matrix = ground_truth::assemble_full_kernel_matrix(params, matr_A, q, QA_cost);
    ground_truth::gemm(alpha, kernel_matrix, B, beta, correct_C);

    // check for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);
    EXPECT_FLOATING_POINT_MATRIX_EQ(C, C2);
}

TYPED_TEST_P(GenericCSVMKernelFunction, PredictValues) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
    }
    if constexpr (kernel == plssvm::kernel_function_type::polynomial || kernel == plssvm::kernel_function_type::sigmoid) {
        params.coef0 = 0.0;
    }

    // create the data that should be used
    const plssvm::aos_matrix<plssvm::real_type> support_vectors{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                   { plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                   { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 } },
                                                                   { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } } } };
    const plssvm::aos_matrix<plssvm::real_type> weights{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                           { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } } };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } };
    plssvm::aos_matrix<plssvm::real_type> w{};
    const plssvm::aos_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 } },
                                                        { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 } } } };

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::rectangular_data_distribution>(plssvm::mpi::communicator{}, data.num_rows(), 1);

    // predict the values using the previously learned support vectors and weights
    const plssvm::aos_matrix<plssvm::real_type> calculated = svm.predict_values(params, support_vectors, weights, rho, w, data);

    // calculate correct predict values
    plssvm::aos_matrix<plssvm::real_type> correct_w;
    if (kernel == plssvm::kernel_function_type::linear) {
        correct_w = ground_truth::calculate_w(weights, support_vectors);
    }
    const plssvm::aos_matrix<plssvm::real_type> correct_predict_values = ground_truth::predict_values(params, correct_w, weights, rho, support_vectors, data);

    // check the calculated result for correctness
    ASSERT_EQ(calculated.shape(), correct_predict_values.shape());
    EXPECT_FLOATING_POINT_MATRIX_NEAR_EPS(calculated, correct_predict_values, 1e6);
    // in case of the linear kernel, the w vector should have been filled
    if (kernel == plssvm::kernel_function_type::linear) {
        EXPECT_EQ(w.num_rows(), rho.size());
        EXPECT_FLOATING_POINT_MATRIX_NEAR(w, (plssvm::aos_matrix<plssvm::real_type>{ weights }));
    } else {
        EXPECT_TRUE(w.empty());
    }
}

TYPED_TEST_P(GenericCSVMKernelFunction, PredictValuesProvidedW) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        SUCCEED() << "Test is only applicable for the linear kernel function!";
    } else {
        // create parameter struct
        plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };  // NOLINT(misc-const-correctness): can change based on the kernel function
        if constexpr (kernel != plssvm::kernel_function_type::linear) {
            params.gamma = plssvm::real_type{ 1.0 };
        }

        // create the data that should be used
        const plssvm::aos_matrix<plssvm::real_type> support_vectors{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                       { plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                       { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 } },
                                                                       { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } } } };
        const plssvm::aos_matrix<plssvm::real_type> weights{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                               { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } } };
        const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } };
        plssvm::aos_matrix<plssvm::real_type> w{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                   { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } } };
        const plssvm::aos_matrix<plssvm::real_type> correct_w{ w };
        const plssvm::aos_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 } },
                                                            { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } } };

        // the correct result
        const plssvm::aos_matrix<plssvm::real_type> correct_predict_values{ { { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                              { plssvm::real_type{ 4.0 }, plssvm::real_type{ 4.0 } } } };

        // create C-SVM: must be done using the mock class since the member function to test is private or protected
        const auto svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

        // be sure to use the correct data distribution
        svm.data_distribution_ = std::make_unique<plssvm::detail::rectangular_data_distribution>(plssvm::mpi::communicator{}, data.num_rows(), 1);

        // predict the values using the previously learned support vectors and weights
        const plssvm::aos_matrix<plssvm::real_type> calculated = svm.predict_values(params, support_vectors, weights, rho, w, data);

        // check the calculated result for correctness
        ASSERT_EQ(calculated.shape(), correct_predict_values.shape());
        EXPECT_FLOATING_POINT_MATRIX_NEAR(calculated, correct_predict_values);
        // in case of the linear kernel, the w vector should not have changed
        EXPECT_FLOATING_POINT_MATRIX_EQ(w, correct_w);
        EXPECT_FLOATING_POINT_MATRIX_NEAR(w, (plssvm::aos_matrix<plssvm::real_type>{ weights }));
    }
}

TYPED_TEST_P(GenericCSVMKernelFunction, PerformDimensionalReduction) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 4 });

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // perform dimensional reduction
    const auto [q_red, QA_cost] = svm.perform_dimensional_reduction(params, data);

    // check values for correctness
    const auto [q_red_correct, QA_cost_correct] = ground_truth::perform_dimensional_reduction(params, data);
    ASSERT_EQ(q_red.size(), data.num_rows() - 1);
    EXPECT_FLOATING_POINT_VECTOR_NEAR(q_red, q_red_correct);
    EXPECT_FLOATING_POINT_NEAR(QA_cost, QA_cost_correct);
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMKernelFunction,
                            BlasLevel3AssemblyImplicitWithoutC,
                            BlasLevel3AssemblyImplicit,
                            PredictValues,
                            PredictValuesProvidedW,
                            PerformDimensionalReduction);

//*************************************************************************************************************************************//
//                                              C-SVM tests depending on the solver type                                               //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMSolver : public GenericCSVM<T> { };

TYPED_TEST_SUITE_P(GenericCSVMSolver);

TYPED_TEST_P(GenericCSVMSolver, SolveLssvmSystemOfLinearEquationsTrivial) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    constexpr plssvm::kernel_function_type kernel = plssvm::kernel_function_type::linear;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create the data that should be used
    // Matrix with 1-1/cost on main diagonal. Thus, the diagonal entries become one with the additional addition of 1/cost
    const plssvm::aos_matrix<plssvm::real_type> A{ { { plssvm::real_type{ std::sqrt(plssvm::real_type{ 1.0 } - (plssvm::real_type{ 1.0 } / params.cost)) }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                     { plssvm::real_type{ 0.0 }, plssvm::real_type{ std::sqrt(plssvm::real_type{ 1.0 } - (plssvm::real_type{ 1.0 } / params.cost)) }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                     { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ std::sqrt(plssvm::real_type{ 1.0 } - (plssvm::real_type{ 1.0 } / params.cost)) }, plssvm::real_type{ 0.0 } },
                                                     { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ std::sqrt(plssvm::real_type{ 1.0 } - (plssvm::real_type{ 1.0 } / params.cost)) } } } };
    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                     { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } } };

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // solve the system of linear equations using the CG algorithm:
    // | Q  1 |  *  | x |  =  | y |
    // | 1  0 |     | b |     | 0 |
    // with Q = A^TA
    const auto &[calculated_x, calculated_rho, num_iters] = svm.solve_lssvm_system_of_linear_equations(A, B, params, plssvm::epsilon = 0.00001, plssvm::solver = solver);

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR_EPS(calculated_x, (plssvm::aos_matrix<plssvm::real_type>{ B }), 1e6);
    EXPECT_TRUE(std::all_of(calculated_rho.cbegin(), calculated_rho.cend(), [front = std::abs(calculated_rho.front())](const plssvm::real_type rho) {
        return std::abs(std::abs(rho) - front) <= std::numeric_limits<plssvm::real_type>::epsilon();
    }));
    for (const auto rho : calculated_rho) {
        EXPECT_LE(std::abs(rho) - std::numeric_limits<plssvm::real_type>::epsilon(), std::numeric_limits<plssvm::real_type>::epsilon());
    }
    EXPECT_THAT(num_iters, ::testing::Each(::testing::Gt(0)));
}

TYPED_TEST_P(GenericCSVMSolver, SolveLssvmSystemOfLinearEquations) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    constexpr plssvm::kernel_function_type kernel = plssvm::kernel_function_type::linear;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 1.0 };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create the data that should be used
    const auto A = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                     { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } } };

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // solve the system of linear equations using the CG algorithm:
    // | Q  1 |  *  | x |  =  | y |
    // | 1  0 |     | b |     | 0 |
    // with Q = A^TA
    const auto &[calculated_x, calculated_rho, num_iters] = svm.solve_lssvm_system_of_linear_equations(A, B, params, plssvm::epsilon = 0.00001, plssvm::solver = solver);

    const plssvm::aos_matrix<plssvm::real_type> correct_x{ { { plssvm::real_type{ 0.4285714285714278 }, plssvm::real_type{ -1.1904761904761898 }, plssvm::real_type{ 1.1904761904761898 }, plssvm::real_type{ -0.4285714285714278 } },
                                                             { plssvm::real_type{ -0.4285714285714278 }, plssvm::real_type{ 1.1904761904761898 }, plssvm::real_type{ -1.1904761904761898 }, plssvm::real_type{ 0.4285714285714278 } } } };

    // check the calculated result for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR_EPS(calculated_x, correct_x, 1e6);  // due to hand provided results
    for (const auto rho : calculated_rho) {
        const auto rho_abs = std::abs(rho);
        const auto calculated_rho_abs = std::abs(calculated_rho.front());
        EXPECT_FLOATING_POINT_NEAR_EPS(rho_abs, calculated_rho_abs, 1e6);  // due to hand provided results
    }
    EXPECT_THAT(num_iters, ::testing::Each(::testing::Gt(0)));
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolver,
                            SolveLssvmSystemOfLinearEquationsTrivial,
                            SolveLssvmSystemOfLinearEquations);

//*************************************************************************************************************************************//
//                                    C-SVM tests depending on the solver and kernel function type                                     //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMSolverKernelFunction : public GenericCSVM<T> { };

TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunction);

TYPED_TEST_P(GenericCSVMSolverKernelFunction, AssembleKernelMatrixMinimal) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 1.0 };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
    }
    if constexpr (kernel == plssvm::kernel_function_type::polynomial || kernel == plssvm::kernel_function_type::sigmoid) {
        params.coef0 = 0.0;
    }
    const plssvm::aos_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                        { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                        { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } } };  // 3 x 3 -> assemble 2 x 2 matrix
    [[maybe_unused]] const std::vector<plssvm::real_type> q_red(data.num_cols() - 1, plssvm::real_type{ 0.0 });
    [[maybe_unused]] const plssvm::real_type QA_cost{ 0.0 };

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();
    // be sure to use the correct data distribution
    const plssvm::detail::triangular_data_distribution dist{ plssvm::mpi::communicator{}, data.num_rows() - 1, num_devices };
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(dist);

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ENABLE_ASSERTS) && defined(PLSSVM_DEATH_TESTS_ENABLED)
        EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(solver, params, data, q_red, QA_cost), "An explicit solver type must be provided instead of solver_type::automatic!");
#else
        SUCCEED() << "Solver type is automatic, but assertions are disabled!";
#endif
    } else if constexpr (solver == plssvm::solver_type::cg_explicit) {
        // run the assemble the kernel matrix kernels
        const std::vector<plssvm::detail::move_only_any> kernel_matrix_d = svm.assemble_kernel_matrix(solver, params, data, q_red, QA_cost);
        ASSERT_EQ(kernel_matrix_d.size(), num_devices);

        for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
            SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

            // can't use a move_only_any_cast, if the kernel matrix is empty
            if (svm.data_distribution_->place_specific_num_rows(device_id) == 0) {
                // the move_only_any shouldn't hold any value!
                EXPECT_FALSE(kernel_matrix_d[device_id].has_value());
            } else {
                EXPECT_TRUE(kernel_matrix_d[device_id].has_value());

                // calculate the ground truth
                const std::vector<plssvm::real_type> correct_partial_kernel_matrix = ground_truth::assemble_device_specific_kernel_matrix(params, data, q_red, QA_cost, *svm.data_distribution_, device_id);

                // get result based on used backend
                std::vector<plssvm::real_type> kernel_matrix{};
                if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::stdpar || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::hpx) {
                    const auto &kernel_matrix_d_ptr = plssvm::detail::move_only_any_cast<const std::unique_ptr<plssvm::real_type[]> &>(kernel_matrix_d[device_id]);  // NOLINT: C-style array must be used here to align with real implementation
                    kernel_matrix.resize(dist.calculate_explicit_kernel_matrix_num_entries(0));
                    std::memcpy(kernel_matrix.data(), kernel_matrix_d_ptr.get(), kernel_matrix.size() * sizeof(plssvm::real_type));
                } else {
                    const auto &kernel_matrix_d_ptr = plssvm::detail::move_only_any_cast<const device_ptr_type &>(kernel_matrix_d[device_id]);  // device_ptr -> convert it to a std::vector
                    kernel_matrix.resize(kernel_matrix_d_ptr.size());
                    kernel_matrix_d_ptr.copy_to_host(kernel_matrix);
                }

                EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, correct_partial_kernel_matrix);
            }
        }
    } else if constexpr (solver == plssvm::solver_type::cg_implicit) {
        // run the assemble the kernel matrix kernels
        const std::vector<plssvm::detail::move_only_any> kernel_matrix_d = svm.assemble_kernel_matrix(solver, params, data, q_red, QA_cost);
        ASSERT_EQ(kernel_matrix_d.size(), num_devices);

        for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
            SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

            // can't use a move_only_any_cast, if the kernel matrix is empty
            if (svm.data_distribution_->place_specific_num_rows(device_id) == 0) {
                // the move_only_any shouldn't hold any value!
                EXPECT_FALSE(kernel_matrix_d[device_id].has_value());
            } else {
                EXPECT_TRUE(kernel_matrix_d[device_id].has_value());

                // implicit doesn't assemble a kernel matrix!
                if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::stdpar || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::hpx) {
                    const auto &[data_d_ret, params_ret, q_red_ret, QA_cost_ret] = plssvm::detail::move_only_any_cast<const std::tuple<const plssvm::aos_matrix<plssvm::real_type> &, plssvm::parameter, const std::vector<plssvm::real_type> &, plssvm::real_type> &>(kernel_matrix_d[device_id]);

                    // the values should not have changed! (except the matrix layout)
                    EXPECT_EQ(params_ret, params);
                    EXPECT_FLOATING_POINT_MATRIX_EQ(data_d_ret, data);
                    EXPECT_EQ(q_red_ret, q_red);
                    EXPECT_EQ(QA_cost_ret, QA_cost);
                } else {
                    const auto &[exec, data_d_ret, params_ret, q_red_d_ret, QA_cost_ret] = plssvm::detail::move_only_any_cast<const std::tuple<plssvm::detail::execution_range, device_ptr_type, plssvm::parameter, device_ptr_type, plssvm::real_type> &>(kernel_matrix_d[device_id]);

                    // copy data back to host
                    plssvm::aos_matrix<plssvm::real_type> data_ret{ data };
                    data_d_ret.copy_to_host(data_ret);
                    std::vector<plssvm::real_type> q_red_ret(q_red.size());
                    q_red_d_ret.copy_to_host(q_red_ret, 0, q_red_ret.size());

                    // the values should not have changed! (except the matrix layout)
                    EXPECT_EQ(params_ret, params);
                    EXPECT_FLOATING_POINT_MATRIX_EQ(data_ret, data);
                    EXPECT_EQ(q_red_ret, q_red);
                    EXPECT_EQ(QA_cost_ret, QA_cost);
                }
            }
        }
    }
}

TYPED_TEST_P(GenericCSVMSolverKernelFunction, AssembleKernelMatrix) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can change based on the kernel function
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 / 3.0 };
    }
    if constexpr (kernel == plssvm::kernel_function_type::polynomial || kernel == plssvm::kernel_function_type::sigmoid) {
        params.coef0 = 1.0;
    }
    const plssvm::aos_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                        { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                        { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } } };
    [[maybe_unused]] const std::vector<plssvm::real_type> q_red = { plssvm::real_type{ 3.0 }, plssvm::real_type{ 4.0 } };
    [[maybe_unused]] const plssvm::real_type QA_cost{ 2.0 };

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();
    // be sure to use the correct data distribution
    const plssvm::detail::triangular_data_distribution dist{ plssvm::mpi::communicator{}, data.num_rows() - 1, num_devices };
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(dist);

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ENABLE_ASSERTS) && defined(PLSSVM_DEATH_TESTS_ENABLED)
        EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(solver, params, data, q_red, QA_cost), "An explicit solver type must be provided instead of solver_type::automatic!");
#else
        SUCCEED() << "Solver type is automatic, but assertions are disabled!";
#endif
    } else if constexpr (solver == plssvm::solver_type::cg_explicit) {
        // run the assemble the kernel matrix kernels
        const std::vector<plssvm::detail::move_only_any> kernel_matrix_d = svm.assemble_kernel_matrix(solver, params, data, q_red, QA_cost);
        ASSERT_EQ(kernel_matrix_d.size(), num_devices);

        for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
            SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

            // can't use a move_only_any_cast, if the kernel matrix is empty
            if (svm.data_distribution_->place_specific_num_rows(device_id) == 0) {
                // the move_only_any shouldn't hold any value!
                EXPECT_FALSE(kernel_matrix_d[device_id].has_value());
            } else {
                EXPECT_TRUE(kernel_matrix_d[device_id].has_value());

                // calculate the ground truth
                const std::vector<plssvm::real_type> correct_partial_kernel_matrix = ground_truth::assemble_device_specific_kernel_matrix(params, data, q_red, QA_cost, *svm.data_distribution_, device_id);

                // get result based on used backend
                std::vector<plssvm::real_type> kernel_matrix{};
                if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::stdpar || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::hpx) {
                    const auto &kernel_matrix_d_ptr = plssvm::detail::move_only_any_cast<const std::unique_ptr<plssvm::real_type[]> &>(kernel_matrix_d[device_id]);  // NOLINT: C-style array must be used here to align with real implementation
                    kernel_matrix.resize(dist.calculate_explicit_kernel_matrix_num_entries(0));
                    std::memcpy(kernel_matrix.data(), kernel_matrix_d_ptr.get(), kernel_matrix.size() * sizeof(plssvm::real_type));
                } else {
                    const auto &kernel_matrix_d_ptr = plssvm::detail::move_only_any_cast<const device_ptr_type &>(kernel_matrix_d[device_id]);  // device_ptr -> convert it to a std::vector
                    kernel_matrix.resize(kernel_matrix_d_ptr.size());
                    kernel_matrix_d_ptr.copy_to_host(kernel_matrix);
                }

                EXPECT_FLOATING_POINT_VECTOR_NEAR(kernel_matrix, correct_partial_kernel_matrix);
            }
        }
    } else if constexpr (solver == plssvm::solver_type::cg_implicit) {
        // run the assemble the kernel matrix kernels
        const std::vector<plssvm::detail::move_only_any> kernel_matrix_d = svm.assemble_kernel_matrix(solver, params, data, q_red, QA_cost);
        ASSERT_EQ(kernel_matrix_d.size(), num_devices);

        for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
            SCOPED_TRACE(fmt::format("device_id {} ({}/{})", device_id, device_id + 1, num_devices));

            // can't use a move_only_any_cast, if the kernel matrix is empty
            if (svm.data_distribution_->place_specific_num_rows(device_id) == 0) {
                // the move_only_any shouldn't hold any value!
                EXPECT_FALSE(kernel_matrix_d[device_id].has_value());
            } else {
                EXPECT_TRUE(kernel_matrix_d[device_id].has_value());

                // implicit doesn't assemble a kernel matrix!
                if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::stdpar || plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::hpx) {
                    const auto &[data_d_ret, params_ret, q_red_ret, QA_cost_ret] = plssvm::detail::move_only_any_cast<const std::tuple<const plssvm::aos_matrix<plssvm::real_type> &, plssvm::parameter, const std::vector<plssvm::real_type> &, plssvm::real_type> &>(kernel_matrix_d[device_id]);

                    // the values should not have changed! (except the matrix layout)
                    EXPECT_EQ(params_ret, params);
                    EXPECT_FLOATING_POINT_MATRIX_EQ(data_d_ret, data);
                    EXPECT_EQ(q_red_ret, q_red);
                    EXPECT_EQ(QA_cost_ret, QA_cost);
                } else {
                    const auto &[exec, data_d_ret, params_ret, q_red_d_ret, QA_cost_ret] = plssvm::detail::move_only_any_cast<const std::tuple<plssvm::detail::execution_range, device_ptr_type, plssvm::parameter, device_ptr_type, plssvm::real_type> &>(kernel_matrix_d[device_id]);

                    // copy data back to host
                    plssvm::aos_matrix<plssvm::real_type> data_ret{ data };
                    data_d_ret.copy_to_host(data_ret);
                    std::vector<plssvm::real_type> q_red_ret(q_red.size());
                    q_red_d_ret.copy_to_host(q_red_ret, 0, q_red_ret.size());

                    // the values should not have changed! (except the matrix layout)
                    EXPECT_EQ(params_ret, params);
                    EXPECT_FLOATING_POINT_MATRIX_EQ(data_ret, data);
                    EXPECT_EQ(q_red_ret, q_red);
                    EXPECT_EQ(QA_cost_ret, QA_cost);
                }
            }
        }
    }
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunction,
                            AssembleKernelMatrixMinimal,
                            AssembleKernelMatrix);

//*************************************************************************************************************************************//
//                                                          C-SVM DeathTests                                                           //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMDeathTest : public GenericCSVM<T> { };

TYPED_TEST_SUITE_P(GenericCSVMDeathTest);

TYPED_TEST_P(GenericCSVMDeathTest, BlasLevel3Automatic) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    constexpr plssvm::solver_type solver = plssvm::solver_type::automatic;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const auto svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    const plssvm::real_type alpha{ 1.0 };

    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 } },
          { plssvm::real_type{ 0.2 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 } },
          { plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm) };

    const plssvm::aos_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } } };

    const plssvm::real_type beta{ 0.0 };
    plssvm::aos_matrix<plssvm::real_type> C{ plssvm::shape{ 3, 3 } };

    // automatic solver type not permitted
    EXPECT_DEATH(svm.blas_level_3(solver, alpha, A, B, beta, C), "An explicit solver type must be provided instead of solver_type::automatic!");
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMDeathTest,
                            BlasLevel3Automatic);

template <typename T>
class GenericCSVMSolverDeathTest : public GenericCSVM<T> { };

TYPED_TEST_SUITE_P(GenericCSVMSolverDeathTest);

TYPED_TEST_P(GenericCSVMSolverDeathTest, ConjugateGradientsEmptyB) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    const plssvm::aos_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{ plssvm::gamma = plssvm::real_type{ 1.0 } };
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_matr{};

    EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, empty_matr, plssvm::real_type{ 0.001 }, 6, solver), "The right-hand sides must not be empty!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, ConjugateGradientsInvalidEps) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    const plssvm::aos_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{ plssvm::gamma = plssvm::real_type{ 1.0 } };
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 6 } };

    EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, B, 0.0, 6, solver), "The epsilon value must be greater than 0.0!");
    EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, B, -0.5, 6, solver), "The epsilon value must be greater than 0.0!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, ConjugateGradientsInvalidMaxCgIter) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    const plssvm::aos_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{ plssvm::gamma = plssvm::real_type{ 1.0 } };
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 6 } };

    EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.001 }, 0, solver), "The maximum number of iterations must be greater than 0!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, RunBlasLevel3WrongNumberOfKernelMatrixParts) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    if constexpr (solver == plssvm::solver_type::automatic) {
        SUCCEED() << "Test not applicable for the automatic solver type!";
    } else {
        // create C-SVM: must be done using the mock class since the member function to test is private or protected
        const mock_csvm_type svm{};

        const plssvm::aos_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
        // parameter necessary for cg_implicit
        const plssvm::parameter params{ plssvm::gamma = plssvm::real_type{ 1.0 } };
        const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
        const plssvm::real_type QA_cost{ 1.0 };

        // be sure to use the correct data distribution
        svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
        std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };
        A.pop_back();

        const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 4, 4 } };
        plssvm::aos_matrix<plssvm::real_type> C{ plssvm::shape{ 4, 4 } };

        EXPECT_DEATH(std::ignore = svm.run_blas_level_3(solver, plssvm::real_type{ 1.0 }, A, B, plssvm::real_type{ 1.0 }, C),
                     ::testing::HasSubstr(fmt::format("Not enough kernel matrix parts ({}) for the available number of devices ({})!", A.size(), svm.num_available_devices())));
    }
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, BlasLevel3EmptyMatrices) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    if constexpr (solver == plssvm::solver_type::automatic) {
        SUCCEED() << "Test not applicable for the automatic solver type!";
    } else {
        // create C-SVM: must be done using the mock class since the member function to test is private or protected
        const mock_csvm_type svm{};

        const plssvm::aos_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
        // parameter necessary for cg_implicit
        const plssvm::parameter params{ plssvm::gamma = plssvm::real_type{ 1.0 } };
        const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
        const plssvm::real_type QA_cost{ 1.0 };

        // be sure to use the correct data distribution
        svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
        const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

        plssvm::aos_matrix<plssvm::real_type> matr{ plssvm::shape{ 4, 4 } };
        plssvm::aos_matrix<plssvm::real_type> empty_matr{};

        EXPECT_DEATH(svm.blas_level_3(solver, plssvm::real_type{ 1.0 }, A, empty_matr, plssvm::real_type{ 1.0 }, matr), "The B matrix must not be empty!");
        EXPECT_DEATH(svm.blas_level_3(solver, plssvm::real_type{ 1.0 }, A, matr, plssvm::real_type{ 1.0 }, empty_matr), "The C matrix must not be empty!");
    }
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, BlasLevel3MatrixShapeMismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    if constexpr (solver == plssvm::solver_type::automatic) {
        SUCCEED() << "Test not applicable for the automatic solver type!";
    } else {
        // create C-SVM: must be done using the mock class since the member function to test is private or protected
        const mock_csvm_type svm{};

        const plssvm::aos_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
        // parameter necessary for cg_implicit
        const plssvm::parameter params{ plssvm::gamma = plssvm::real_type{ 1.0 } };
        const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
        const plssvm::real_type QA_cost{ 1.0 };

        // be sure to use the correct data distribution
        svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, matr_A.num_rows(), svm.num_available_devices());
        const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

        const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 4, 4 } };
        plssvm::aos_matrix<plssvm::real_type> C{ plssvm::shape{ 3, 3 } };

        EXPECT_DEATH(svm.blas_level_3(solver, plssvm::real_type{ 1.0 }, A, B, plssvm::real_type{ 1.0 }, C),
                     ::testing::HasSubstr("The B ([4, 4]) and C ([3, 3]) matrices must have the same shape!"));
    }
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolverDeathTest,
                            ConjugateGradientsEmptyB,
                            ConjugateGradientsInvalidEps,
                            ConjugateGradientsInvalidMaxCgIter,
                            RunBlasLevel3WrongNumberOfKernelMatrixParts,
                            BlasLevel3EmptyMatrices,
                            BlasLevel3MatrixShapeMismatch);

template <typename T>
class GenericCSVMKernelFunctionDeathTest : public GenericCSVMKernelFunction<T> { };

TYPED_TEST_SUITE_P(GenericCSVMKernelFunctionDeathTest);

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, SolveLssvmSystemOfLinearEquationsEmptyA) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_matr{};
    const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 4 } };

    EXPECT_DEATH(std::ignore = svm.solve_lssvm_system_of_linear_equations(empty_matr, B, params), "The A matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, SolveLssvmSystemOfLinearEquationsEmptyB) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_matr{};
    const plssvm::aos_matrix<plssvm::real_type> A{ plssvm::shape{ 6, 4 } };

    EXPECT_DEATH(std::ignore = svm.solve_lssvm_system_of_linear_equations(A, empty_matr, params), "The B matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, SolveLssvmSystemOfLinearEquationsSizeMismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> A{ plssvm::shape{ 6, 4 } };
    const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 3 } };

    EXPECT_DEATH(std::ignore = svm.solve_lssvm_system_of_linear_equations(A, B, params), ::testing::HasSubstr("The number of data points in A (6) and B (3) must be the same!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, PerformDimensionalReductionEmptyA) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_matr{};

    EXPECT_DEATH(std::ignore = svm.perform_dimensional_reduction(params, empty_matr), "The matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, AssembleKernelMatrixAutomatic) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto A = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 5 });
    const std::vector<plssvm::real_type> q_red(A.num_rows() - 1);
    const plssvm::real_type QA_cost = 42.0;

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, A.num_rows() - 1, svm.num_available_devices());

    // the solver type must not be automatic
    EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(plssvm::solver_type::automatic, params, A, q_red, QA_cost), ::testing::HasSubstr("An explicit solver type must be provided instead of solver_type::automatic!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, PredictValuesEmptyMatrices) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_aos_matr{};
    const plssvm::aos_matrix<plssvm::real_type> empty_soa_matr{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });
    const std::vector<plssvm::real_type> rho(2);
    plssvm::aos_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, data.num_rows(), svm.num_available_devices());

    // support vectors shouldn't be empty
    EXPECT_DEATH(std::ignore = svm.predict_values(params, empty_soa_matr, weights, rho, w, data), "The support vectors must not be empty!");
    // weights shouldn't be empty
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, empty_aos_matr, rho, w, data), ::testing::HasSubstr("The alpha vectors (weights) must not be empty!"));
    // predict points shouldn't be empty
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, empty_soa_matr), "The data points to predict must not be empty!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, PredictValuesSvAlphaSizeMismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, 4 });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });
    const std::vector<plssvm::real_type> rho(2);
    plssvm::aos_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, data.num_rows(), svm.num_available_devices());

    // the number of support vectors and weights must be identical
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("The number of support vectors (3) and number of weights (4) must be the same!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, PredictValuesRhoAlphaSizeMismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });
    const std::vector<plssvm::real_type> rho(1);
    plssvm::aos_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, data.num_rows(), svm.num_available_devices());

    // the number of rho values and weight vectors must be identical
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("The number of rho values (1) and the number of weight vectors (2) must be the same!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, PredictValuesWSizeMismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });
    const std::vector<plssvm::real_type> rho(2);
    const auto data = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, data.num_rows(), svm.num_available_devices());

    // the number of features and w values must be identical
    auto w = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 3 });
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("Either w must be empty or contain exactly the same number of values (3) as features are present (4)!"));
    // the number of weight and w vectors must be identical
    w = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 3, 4 });
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("Either w must be empty or contain exactly the same number of vectors (3) as the alpha vector (2)!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, PredictValuesNumFeaturesMismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can't be const for the chi-squared kernel
    if constexpr (kernel != plssvm::kernel_function_type::linear) {
        params.gamma = plssvm::real_type{ 1.0 };
    }

    // create C-SVM: must be done using the mock class since the member function to test is private or protected
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 5 });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });
    const std::vector<plssvm::real_type> rho(2);
    plssvm::aos_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, data.num_rows(), svm.num_available_devices());

    // the number of features for the support vectors and predict points must be identical
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("The number of features in the support vectors (5) must be the same as in the data points to predict (4)!"));
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMKernelFunctionDeathTest,
                            SolveLssvmSystemOfLinearEquationsEmptyA,
                            SolveLssvmSystemOfLinearEquationsEmptyB,
                            SolveLssvmSystemOfLinearEquationsSizeMismatch,
                            PerformDimensionalReductionEmptyA,
                            AssembleKernelMatrixAutomatic,
                            PredictValuesEmptyMatrices,
                            PredictValuesSvAlphaSizeMismatch,
                            PredictValuesRhoAlphaSizeMismatch,
                            PredictValuesWSizeMismatch,
                            PredictValuesNumFeaturesMismatch);

template <typename T>
class GenericCSVMSolverKernelFunctionDeathTest : public GenericCSVMSolverKernelFunction<T> { };

TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunctionDeathTest);

TYPED_TEST_P(GenericCSVMSolverKernelFunctionDeathTest, AssembleKernelMatrixEmptyMatrices) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    [[maybe_unused]] constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    if constexpr (solver == plssvm::solver_type::automatic) {
        SUCCEED() << "Test not applicable for the automatic solver type!";
    } else {
        // create parameter
        plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can change based on the kernel function
        if constexpr (kernel != plssvm::kernel_function_type::linear) {
            params.gamma = plssvm::real_type{ 1.0 };
        }

        // create C-SVM: must be done using the mock class since the member function to test is private or protected
        const mock_csvm_type svm{};

        // create correct input matrices
        const auto A = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 5 });
        const std::vector<plssvm::real_type> q_red(A.num_rows() - 1);
        const plssvm::real_type QA_cost = 42.0;

        // be sure to use the correct data distribution
        svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, A.num_rows() - 1, svm.num_available_devices());

        const plssvm::aos_matrix<plssvm::real_type> empty_matr{};
        const std::vector<plssvm::real_type> empty_vec{};

        // the A matrix must not be empty
        EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(solver, params, empty_matr, q_red, QA_cost), ::testing::HasSubstr("The matrix to setup on the devices must not be empty!"));
        // the q_red vector must not be empty
        EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(solver, params, A, empty_vec, QA_cost), ::testing::HasSubstr("The q_red vector must not be empty!"));
    }
}

TYPED_TEST_P(GenericCSVMSolverKernelFunctionDeathTest, AssembleKernelMatrixSizeMismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    [[maybe_unused]] constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    if constexpr (solver == plssvm::solver_type::automatic) {
        SUCCEED() << "Test not applicable for the automatic solver type!";
    } else {
        // create parameter
        plssvm::parameter params{ plssvm::kernel_type = kernel };  // NOLINT(misc-const-correctness): can change based on the kernel function
        if constexpr (kernel != plssvm::kernel_function_type::linear) {
            params.gamma = plssvm::real_type{ 1.0 };
        }

        // create C-SVM: must be done using the mock class since the member function to test is private or protected
        const mock_csvm_type svm{};

        // create correct input matrices
        const auto A = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 5 });
        const std::vector<plssvm::real_type> q_red(A.num_rows());
        const plssvm::real_type QA_cost = 42.0;

        // be sure to use the correct data distribution
        svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(plssvm::mpi::communicator{}, A.num_rows() - 1, svm.num_available_devices());

        // the sizes must match
        EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(solver, params, A, q_red, QA_cost), ::testing::HasSubstr("The q_red size (4) mismatches the number of data points after dimensional reduction (3)!"));
    }
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunctionDeathTest,
                            AssembleKernelMatrixEmptyMatrices,
                            AssembleKernelMatrixSizeMismatch);

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_BASE_CSVM_TESTS_HPP_
