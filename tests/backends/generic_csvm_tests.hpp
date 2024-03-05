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
#include "plssvm/detail/data_distribution.hpp"  // plssvm::detail::{triangular_data_distribution, rectangular_data_distribution}
#include "plssvm/detail/io/file_reader.hpp"     // plssvm::detail::io::file_reader
#include "plssvm/detail/memory_size.hpp"        // memory size literals
#include "plssvm/detail/move_only_any.hpp"      // plssvm::detail::move_only_any
#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/detail/utility.hpp"            // plssvm::detail::{unreachable, get, unreachable}
#include "plssvm/kernel_function_types.hpp"     // plssvm::csvm_to_backend_type_v, plssvm::backend_type
#include "plssvm/matrix.hpp"                    // plssvm::aos_matrix, plssvm::layout_type
#include "plssvm/model.hpp"                     // plssvm::model
#include "plssvm/parameter.hpp"                 // plssvm::parameter
#include "plssvm/shape.hpp"                     // plssvm::shape
#include "plssvm/solver_types.hpp"              // plssvm::solver_type
#include "plssvm/target_platforms.hpp"          // plssvm::target_platform

#include "tests/backends/ground_truth.hpp"  // ground_truth::{kernel_function, perform_dimensional_reduction}
#include "tests/custom_test_macros.hpp"     // EXPECT_FLOATING_POINT_MATRIX_EQ, EXPECT_FLOATING_POINT_VECTOR_NEAR, EXPECT_FLOATING_POINT_NEAR
#include "tests/types_to_test.hpp"          // util::{test_parameter_type_at_t, test_parameter_value_at_v}
#include "tests/utility.hpp"                // util::{redirect_output, generate_specific_matrix, construct_from_tuple, flatten, generate_random_matrix}

#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads
#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TYPED_TEST_SUITE_P, TYPED_TEST_P, REGISTER_TYPED_TEST_SUITE_P, EXPECT_EQ, EXPECT_NE, EXPECT_GT, EXPECT_TRUE, EXPECT_DEATH,
                          // ASSERT_EQ, GTEST_SKIP, SUCCEED, ::testing::Test

#include <array>    // std::array
#include <cmath>    // std::sqrt, std::abs, std::exp, std::pow
#include <cstddef>  // std::size_t
#include <limits>   // std::numeric_limits::epsilon
#include <memory>   // std::unique_ptr, std::make_unique
#include <tuple>    // std::ignore, std::tuple, std::make_tuple
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
[[nodiscard]] inline std::vector<T> pad_1D_vector(std::vector<T> matr, [[maybe_unused]] const plssvm::shape shape) {
    // the OpenMP backend has no padding!
    if constexpr (plssvm::csvm_to_backend_type_v<plssvm_csvm_type> != plssvm::backend_type::openmp) {
        std::size_t idx{ matr.size() };
        for (std::size_t i = 0; i < shape.x; ++i) {
            matr.insert(matr.cbegin() + idx, plssvm::PADDING_SIZE, plssvm::real_type{ 0.0 });
            idx -= i + 1;
        }
        matr.insert(matr.cend(), plssvm::PADDING_SIZE * (plssvm::PADDING_SIZE + 1) / 2, plssvm::real_type{ 0.0 });
    }
    return matr;
}

/**
 * @brief Initialize the explicit kernel matrices adding necessary padding entries using the values provided by @p matr based on the @p csvm_type.
 * @tparam csvm_type the PLSSVM CSVM backend type
 * @tparam device_ptr_type the device pointer type; only used if a GPU backend is used
 * @tparam matrix_type the type of the matrix
 * @tparam used_csvm_type the type of the @p csvm; may also be a Mock CSVM!
 * @tparam Args the types of the additional arguments; none for the explicit matrix
 * @param[in] matr the matrix to create the explicit kernel matrix from
 * @param[in] csvm the CSVM encapsulating the device on which the matrix should be allocated
 * @return a std::vector of `plssvm::detail::move_only_any` with a wrapped value usable in the PLSSVM functions (`[[nodiscard]]`)
 */
template <typename csvm_type, typename device_ptr_type, typename matrix_type, typename used_csvm_type, typename... Args>
[[nodiscard]] inline std::vector<plssvm::detail::move_only_any> init_explicit_matrices(matrix_type matr, used_csvm_type &csvm) {
    using real_type = typename matrix_type::value_type;
    std::vector<plssvm::detail::move_only_any> result(csvm.num_available_devices());

    for (std::size_t device_id = 0; device_id < csvm.num_available_devices(); ++device_id) {
        auto &device = csvm.devices_[device_id];
        const std::size_t device_specific_num_rows = csvm.data_distribution_->place_specific_num_rows(device_id);
        const std::size_t row_offset = csvm.data_distribution_->place_row_offset(device_id);

        // split kernel matrix on the cpu
        std::vector<real_type> partial_kernel_matrix;
        for (std::size_t row = row_offset; row < row_offset + device_specific_num_rows; ++row) {
            for (std::size_t col = row; col < matr.num_cols(); ++col) {
                partial_kernel_matrix.push_back(matr(row, col));
            }
            // add padding
            partial_kernel_matrix.insert(partial_kernel_matrix.cend(), plssvm::PADDING_SIZE, real_type{ 0.0 });
        }
        // add padding
        const std::size_t remaining_rows = matr.num_rows() - (row_offset + device_specific_num_rows);
        const std::size_t remaining_rows_without_padding = remaining_rows - plssvm::PADDING_SIZE;
        const std::size_t num_padding_entries = (remaining_rows * (remaining_rows + 1) / 2) - (remaining_rows_without_padding * (remaining_rows_without_padding + 1) / 2);
        partial_kernel_matrix.insert(partial_kernel_matrix.cend(), num_padding_entries + static_cast<std::size_t>(plssvm::PADDING_SIZE * plssvm::PADDING_SIZE), real_type{ 0.0 });

        // move kernel matrix to the specific device
        if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp) {
            result.push_back(plssvm::detail::move_only_any{ partial_kernel_matrix });  // TODO: OpenMP padding?
        } else {
            device_ptr_type ptr{ partial_kernel_matrix.size(), device };
            ptr.copy_to_device(partial_kernel_matrix);
            result[device_id] = plssvm::detail::move_only_any{ std::move(ptr) };
        }
    }
    return result;
}

/**
 * @brief Initialize the implicit kernel matrices adding necessary padding entries using the values provided by @p matr based on the @p csvm_type.
 * @tparam csvm_type the PLSSVM CSVM backend type
 * @tparam device_ptr_type the device pointer type; only used if a GPU backend is used
 * @tparam matrix_type the type of the matrix
 * @tparam used_csvm_type the type of the @p csvm; may also be a Mock CSVM!
 * @tparam Args the types of the additional arguments
 * @param[in] matr the matrix to create the explicit kernel matrix from
 * @param[in] csvm the CSVM encapsulating the device on which the matrix should be allocated
 * @param[in] args the additional arguments
 * @return a std::vector of `plssvm::detail::move_only_any` with a wrapped value usable in the PLSSVM functions (`[[nodiscard]]`)
 */
template <typename csvm_type, typename device_ptr_type, typename matrix_type, typename used_csvm_type, typename... Args>
[[nodiscard]] inline std::vector<plssvm::detail::move_only_any> init_implicit_matrices(matrix_type matr, used_csvm_type &csvm, Args &&...args) {
    using real_type = typename matrix_type::value_type;
    std::vector<plssvm::detail::move_only_any> result(csvm.num_available_devices());

    const auto params = static_cast<plssvm::parameter>(plssvm::detail::get<0>(args...));
    const auto q_red = static_cast<std::vector<real_type>>(plssvm::detail::get<1>(args...));
    const auto QA_cost = static_cast<real_type>(plssvm::detail::get<2>(args...));

    // add padding to the input matrix
    matr = matrix_type{ matr, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    for (std::size_t device_id = 0; device_id < csvm.num_available_devices(); ++device_id) {
        // created matrix is different for the OpenMP backend and the GPU backends!
        if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp) {
            // only a single device ever in use
            result[0] = plssvm::detail::move_only_any{ std::make_tuple(plssvm::aos_matrix<real_type>{ matr, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } }, std::forward<Args>(args)...) };
        } else {
            auto &device = csvm.devices_[device_id];

            // create device pointer
            device_ptr_type matr_d{ matr.shape(), matr.padding(), device };
            matr_d.copy_to_device(matr);

            // create device pointer for the q vector
            device_ptr_type q_d{ q_red.size() + plssvm::PADDING_SIZE, device };
            q_d.copy_to_device(q_red, 0, q_red.size());
            q_d.memset(0, q_red.size());

            result[device_id] = plssvm::detail::move_only_any{ std::make_tuple(std::move(matr_d), params, std::move(q_d), QA_cost) };
        }
    }
    return result;
}

/**
 * @brief Initialize the kernel matrices based on the used @p solver type adding necessary padding entries using the values provided by @p matr based on the @p csvm_type.
 * @tparam csvm_type the PLSSVM CSVM backend type
 * @tparam device_ptr_type the device pointer type; only used if a GPU backend is used
 * @tparam matrix_type the type of the matrix
 * @tparam used_csvm_type the type of the @p csvm; may also be a Mock CSVM!
 * @tparam Args the types of the additional arguments
 * @param[in] matr the matrix to create the explicit kernel matrix from
 * @param[in] solver the used solver type in the CG algorithm
 * @param[in] csvm the CSVM encapsulating the device on which the matrix should be allocated
 * @param[in] args the additional arguments
 * @return a std::vector of `plssvm::detail::move_only_any` with a wrapped value usable in the PLSSVM functions (`[[nodiscard]]`)
 */
template <typename csvm_type, typename device_ptr_type, typename matrix_type, typename used_csvm_type, typename... Args>
[[nodiscard]] inline std::vector<plssvm::detail::move_only_any> init_matrices(matrix_type matr, const plssvm::solver_type solver, [[maybe_unused]] used_csvm_type &csvm, Args &&...args) {
    switch (solver) {
        case plssvm::solver_type::automatic:
            {
                std::vector<plssvm::detail::move_only_any> result;
                for (std::size_t device_id = 0; device_id < csvm.num_available_devices(); ++device_id) {
                    result.push_back(plssvm::detail::move_only_any{ std::vector<typename matrix_type::value_type>{} });
                }
                return result;  // dummy return only necessary for the DeathTests -> VALUE NOT USED!
            }
        case plssvm::solver_type::cg_explicit:
            // no additional arguments are used
            return init_explicit_matrices<csvm_type, device_ptr_type>(std::move(matr), csvm);
        case plssvm::solver_type::cg_streaming:
            {
                std::vector<plssvm::detail::move_only_any> result;
                for (std::size_t device_id = 0; device_id < csvm.num_available_devices(); ++device_id) {
                    result.push_back(plssvm::detail::move_only_any{ std::vector<typename matrix_type::value_type>{} });
                }
                return result;  // dummy return only necessary for the DeathTests -> VALUE NOT USED!
            }
        case plssvm::solver_type::cg_implicit:
            // additional arguments are: params, q_red, QA_cost
            return init_implicit_matrices<csvm_type, device_ptr_type>(std::move(matr), csvm, std::forward<Args>(args)...);
    }
    // should never be reached!
    plssvm::detail::unreachable();
}

}  // namespace util

//*************************************************************************************************************************************//
//                                                   CSVM tests depending on nothing                                                   //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVM : public ::testing::Test,
                    protected util::redirect_output<> { };

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
    const std::vector<plssvm::detail::memory_size> mem = svm.get_device_memory();
    EXPECT_GE(mem.size(), 1);
    for (const plssvm::detail::memory_size ms : mem) {
        EXPECT_GT(ms, 0_B);
    }
}

TYPED_TEST_P(GenericCSVM, get_max_mem_alloc_size) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the maximum memory allocation size should be greater than 0!
    using namespace plssvm::detail::literals;
    const std::vector<plssvm::detail::memory_size> mem = svm.get_max_mem_alloc_size();
    EXPECT_GE(mem.size(), 1);
    for (const plssvm::detail::memory_size ms : mem) {
        EXPECT_GT(ms, 0_B);
    }
}

TYPED_TEST_P(GenericCSVM, num_available_devices) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;

    // create C-SVM
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // the maximum memory allocation size should be greater than 0!
    using namespace plssvm::detail::literals;
    EXPECT_GE(svm.num_available_devices(), 1);
}

TYPED_TEST_P(GenericCSVM, blas_level_3_explicit_without_C) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_explicit;

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::blas_level_3 is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    [[maybe_unused]] const plssvm::real_type alpha{ 1.0 };

    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 } },
          { plssvm::real_type{ 0.2 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 } },
          { plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ std::move(util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm)) };

    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } },
                                                   plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    [[maybe_unused]] const plssvm::real_type beta{ 0.0 };
    plssvm::soa_matrix<plssvm::real_type> C{ plssvm::shape{ 3, 3 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    plssvm::soa_matrix<plssvm::real_type> C2{ C };

    // perform BLAS calculation
    svm.blas_level_3(solver, alpha, A, B, beta, C);

    // check C for correctness
    const plssvm::soa_matrix<plssvm::real_type> correct_C{ { { plssvm::real_type{ 1.4 }, plssvm::real_type{ 6.5 }, plssvm::real_type{ 9.8 } },
                                                             { plssvm::real_type{ 3.2 }, plssvm::real_type{ 14.6 }, plssvm::real_type{ 21.5 } },
                                                             { plssvm::real_type{ 5.0 }, plssvm::real_type{ 22.7 }, plssvm::real_type{ 33.2 } } },
                                                           plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);

    // check wrapper function
    std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);
    EXPECT_EQ(C2, C);
}

TYPED_TEST_P(GenericCSVM, blas_level_3_streaming_without_C) {
    GTEST_SKIP() << "Currently not implemented!";
}

TYPED_TEST_P(GenericCSVM, blas_level_3_explicit) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_explicit;

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::blas_level_3 is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    [[maybe_unused]] const plssvm::real_type alpha{ 1.0 };

    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 } },
          { plssvm::real_type{ 0.2 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 } },
          { plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm) };

    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } },
                                                   plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    [[maybe_unused]] const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 3, 3 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    plssvm::soa_matrix<plssvm::real_type> C2{ C };

    // perform BLAS calculation
    svm.blas_level_3(solver, alpha, A, B, beta, C);
    std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);

    // check C for correctness
    const plssvm::soa_matrix<plssvm::real_type> correct_C{ { { plssvm::real_type{ 1.45 }, plssvm::real_type{ 6.6 }, plssvm::real_type{ 9.95 } },
                                                             { plssvm::real_type{ 3.75 }, plssvm::real_type{ 15.2 }, plssvm::real_type{ 22.15 } },
                                                             { plssvm::real_type{ 6.05 }, plssvm::real_type{ 23.8 }, plssvm::real_type{ 34.35 } } },
                                                           plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);
    EXPECT_EQ(C2, C);
}

TYPED_TEST_P(GenericCSVM, blas_level_3_streaming) {
    GTEST_SKIP() << "Currently not implemented!";
}

TYPED_TEST_P(GenericCSVM, conjugate_gradients_trivial) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    // solver type doesn't mather since we want to test the surrounding CG algorithm here
    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_explicit;

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::conjugate_gradients is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // create the data that should be used
    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
          { plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
          { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 0.0 } },
          { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm) };

    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                     { plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 } } } };

    // solve AX = B
    const auto [X, num_iter] = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.00001 }, 4, solver);

    // check result
    EXPECT_FLOATING_POINT_MATRIX_NEAR(X, (plssvm::soa_matrix<plssvm::real_type>{ B, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } }));
    EXPECT_GT(num_iter, 0);
}

TYPED_TEST_P(GenericCSVM, conjugate_gradients) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    // solver type doesn't mather since we want to test the surrounding CG algorithm here
    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_explicit;

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::conjugate_gradients is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    // create the data that should be used
    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 4.0 }, plssvm::real_type{ 1.0 } },
          { plssvm::real_type{ 1.0 }, plssvm::real_type{ 3.0 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A = util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm);

    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 } },
                                                     { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 } } } };
    const plssvm::soa_matrix<plssvm::real_type> correct_X{ { { plssvm::real_type{ 1.0 / 11.0 }, plssvm::real_type{ 7.0 / 11.0 } },
                                                             { plssvm::real_type{ 1.0 / 11.0 }, plssvm::real_type{ 7.0 / 11.0 } } },
                                                           plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    // solve AX = B
    const auto [X, num_iters] = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.00001 }, 2, solver);

    // check result
    EXPECT_FLOATING_POINT_MATRIX_NEAR(X, correct_X);
    EXPECT_GT(num_iters, 0);
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVM,
                            move_constructor,
                            move_assignment,
                            get_target_platform,
                            get_device_memory,
                            get_max_mem_alloc_size,
                            num_available_devices,
                            blas_level_3_explicit_without_C,
                            blas_level_3_streaming_without_C,
                            blas_level_3_explicit,
                            blas_level_3_streaming,
                            conjugate_gradients_trivial,
                            conjugate_gradients);

//*************************************************************************************************************************************//
//                                           CSVM tests depending on the kernel function type                                          //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMKernelFunction : public GenericCSVM<T> { };

TYPED_TEST_SUITE_P(GenericCSVMKernelFunction);

TYPED_TEST_P(GenericCSVMKernelFunction, blas_level_3_assembly_implicit_without_C) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // set solver type explicitly to cg_implicit
    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_implicit;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
        params.gamma = 1.0;
        params.coef0 = 0.0;
    } else if constexpr (kernel == plssvm::kernel_function_type::rbf) {
        params.gamma = 1.0;
    }

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::blas_level_3 is protected
    mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    [[maybe_unused]] const plssvm::real_type alpha{ 1.0 };

    const auto matr_A = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const auto [q, QA_cost] = ground_truth::perform_dimensional_reduction(params, matr_A);

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows() - 1, svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_implicit_matrices<csvm_type, device_ptr_type>(matr_A, svm, params, q, QA_cost) };

    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } },
                                                   plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    const plssvm::real_type beta{ 0.0 };
    plssvm::soa_matrix<plssvm::real_type> C{ B.shape(), plssvm::real_type{ 0.0 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    plssvm::soa_matrix<plssvm::real_type> C2{ C };

    // perform BLAS calculation
    svm.blas_level_3(solver, alpha, A, B, beta, C);
    std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);

    // calculate correct results
    const plssvm::aos_matrix<plssvm::real_type> full_kernel_matrix = ground_truth::assemble_full_kernel_matrix(params, matr_A, q, QA_cost);
    plssvm::soa_matrix<plssvm::real_type> correct_C{ C.shape(), plssvm::real_type{ 0.0 }, C.padding() };
    ground_truth::gemm(alpha, full_kernel_matrix, B, beta, correct_C);

    // check for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C2, correct_C);
}

TYPED_TEST_P(GenericCSVMKernelFunction, blas_level_3_assembly_implicit) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // set solver type explicitly to cg_implicit
    constexpr plssvm::solver_type solver = plssvm::solver_type::cg_implicit;

    // create parameter struct
    plssvm::parameter params{ plssvm::kernel_type = kernel, plssvm::cost = 2.0 };
    if constexpr (kernel == plssvm::kernel_function_type::polynomial) {
        params.degree = 1;
        params.gamma = 1.0;
        params.coef0 = 0.0;
    } else if constexpr (kernel == plssvm::kernel_function_type::rbf) {
        params.gamma = 1.0;
    }

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::blas_level_3 is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    [[maybe_unused]] const plssvm::real_type alpha{ 1.0 };

    const auto matr_A = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const auto [q, QA_cost] = ground_truth::perform_dimensional_reduction(params, matr_A);

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows() - 1, svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_implicit_matrices<csvm_type, device_ptr_type>(matr_A, svm, params, q, QA_cost) };

    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } },
                                                   plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    const plssvm::real_type beta{ 0.5 };
    auto C = util::generate_specific_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 3, 3 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    plssvm::soa_matrix<plssvm::real_type> C2{ C };
    plssvm::soa_matrix<plssvm::real_type> correct_C{ C };

    // perform BLAS calculation
    svm.blas_level_3(solver, alpha, A, B, beta, C);
    std::ignore = svm.run_blas_level_3(solver, alpha, A, B, beta, C2);

    // calculate correct results
    const plssvm::aos_matrix<plssvm::real_type> kernel_matrix = ground_truth::assemble_full_kernel_matrix(params, matr_A, q, QA_cost);
    ground_truth::gemm(alpha, kernel_matrix, B, beta, correct_C);

    // check for correctness
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C, correct_C);
    EXPECT_FLOATING_POINT_MATRIX_NEAR(C2, correct_C);
}

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
                                                                 plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    const plssvm::aos_matrix<plssvm::real_type> weights{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                           { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                         plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } };
    plssvm::soa_matrix<plssvm::real_type> w{};
    const plssvm::soa_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 } },
                                                        { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                      plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::predict_values is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::rectangular_data_distribution>(data.num_rows(), 1);

    // predict the values using the previously learned support vectors and weights
    const plssvm::aos_matrix<plssvm::real_type> calculated = svm.predict_values(params, support_vectors, weights, rho, w, data);

    // calculate correct predict values
    plssvm::soa_matrix<plssvm::real_type> correct_w;
    if (kernel == plssvm::kernel_function_type::linear) {
        correct_w = ground_truth::calculate_w(weights, support_vectors);
    }
    const plssvm::aos_matrix<plssvm::real_type> correct_predict_values = ground_truth::predict_values(params, correct_w, weights, rho, support_vectors, data);

    // check the calculated result for correctness
    ASSERT_EQ(calculated.shape(), correct_predict_values.shape());
    EXPECT_FLOATING_POINT_MATRIX_NEAR(calculated, correct_predict_values);
    // in case of the linear kernel, the w vector should have been filled
    if (kernel == plssvm::kernel_function_type::linear) {
        EXPECT_EQ(w.num_rows(), rho.size());
        EXPECT_FLOATING_POINT_MATRIX_NEAR(w, (plssvm::soa_matrix<plssvm::real_type>{ weights, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } }));
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
                                                                     plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
        const plssvm::aos_matrix<plssvm::real_type> weights{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                               { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                             plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
        const std::vector<plssvm::real_type> rho{ plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } };
        plssvm::soa_matrix<plssvm::real_type> w{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } },
                                                   { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                 plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
        const plssvm::soa_matrix<plssvm::real_type> correct_w{ w };
        const plssvm::soa_matrix<plssvm::real_type> data{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ 1.0 } },
                                                            { plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 }, plssvm::real_type{ 1.0 }, plssvm::real_type{ -1.0 } } },
                                                          plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

        // the correct result
        const plssvm::aos_matrix<plssvm::real_type> correct_predict_values{ { { plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 } },
                                                                              { plssvm::real_type{ 4.0 }, plssvm::real_type{ 4.0 } } },
                                                                            plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

        // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::predict_values is protected
        const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

        // be sure to use the correct data distribution
        svm.data_distribution_ = std::make_unique<plssvm::detail::rectangular_data_distribution>(data.num_rows(), 1);

        // predict the values using the previously learned support vectors and weights
        const plssvm::aos_matrix<plssvm::real_type> calculated = svm.predict_values(params, support_vectors, weights, rho, w, data);

        // check the calculated result for correctness
        ASSERT_EQ(calculated.shape(), correct_predict_values.shape());
        EXPECT_FLOATING_POINT_MATRIX_NEAR(calculated, correct_predict_values);
        // in case of the linear kernel, the w vector should not have changed
        EXPECT_FLOATING_POINT_MATRIX_EQ(w, correct_w);
        EXPECT_FLOATING_POINT_MATRIX_NEAR(w, (plssvm::soa_matrix<plssvm::real_type>{ weights, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } }));
    }
}

TYPED_TEST_P(GenericCSVMKernelFunction, perform_dimensional_reduction) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter struct
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 6, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::perform_dimensional_reduction is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);

    // perform dimensional reduction
    const auto [q_red, QA_cost] = svm.perform_dimensional_reduction(params, data);

    // check values for correctness
    const auto [q_red_correct, QA_cost_correct] = ground_truth::perform_dimensional_reduction(params, data);
    ASSERT_EQ(q_red.size(), data.num_rows() - 1);
    EXPECT_FLOATING_POINT_VECTOR_NEAR(q_red, q_red_correct);
    EXPECT_FLOATING_POINT_NEAR(QA_cost, QA_cost_correct);
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMKernelFunction,
                            blas_level_3_assembly_implicit_without_C,
                            blas_level_3_assembly_implicit,
                            predict_values,
                            predict_values_provided_w,
                            perform_dimensional_reduction);

//*************************************************************************************************************************************//
//                                     CSVM tests depending on the solver and kernel function type                                     //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMSolverKernelFunction : public GenericCSVM<T> { };

TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunction);

TYPED_TEST_P(GenericCSVMSolverKernelFunction, solve_lssvm_system_of_linear_equations_trivial) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming) {
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
                                                   plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
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
    EXPECT_FLOATING_POINT_MATRIX_NEAR(calculated_x, (plssvm::aos_matrix<plssvm::real_type>{ B, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } }));
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
    if constexpr (solver == plssvm::solver_type::cg_streaming) {
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
                                                      plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };  // 3 x 3 -> assemble 2 x 2 matrix
    [[maybe_unused]] const std::vector<plssvm::real_type> q_red(data.num_cols() - 1, plssvm::real_type{ 0.0 });
    [[maybe_unused]] const plssvm::real_type QA_cost{ 0.0 };

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::assemble_kernel_matrix is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_rows() - 1, 1);

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ASSERT_ENABLED)
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
                if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp) {
                    kernel_matrix = plssvm::detail::move_only_any_cast<std::vector<plssvm::real_type>>(kernel_matrix_d[device_id]);  // std::vector
                    ASSERT_EQ(kernel_matrix.size(), 3);
                } else {
                    const auto &kernel_matrix_d_ptr = plssvm::detail::move_only_any_cast<const device_ptr_type &>(kernel_matrix_d[device_id]);  // device_ptr -> convert it to a std::vector
                    kernel_matrix.resize(kernel_matrix_d_ptr.size_padded());
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
                if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp) {
                    const auto &[data_d_ret, params_ret, q_red_ret, QA_cost_ret] = plssvm::detail::move_only_any_cast<const std::tuple<plssvm::aos_matrix<plssvm::real_type>, plssvm::parameter, std::vector<plssvm::real_type>, plssvm::real_type> &>(kernel_matrix_d[device_id]);

                    // the values should not have changed! (except the matrix layout)
                    EXPECT_EQ(params_ret, params);
                    EXPECT_FLOATING_POINT_MATRIX_EQ(data_d_ret, plssvm::aos_matrix<plssvm::real_type>{ data });
                    EXPECT_EQ(q_red_ret, q_red);
                    EXPECT_EQ(QA_cost_ret, QA_cost);
                } else {
                    const auto &[data_d_ret, params_ret, q_red_d_ret, QA_cost_ret] = plssvm::detail::move_only_any_cast<const std::tuple<device_ptr_type, plssvm::parameter, device_ptr_type, plssvm::real_type> &>(kernel_matrix_d[device_id]);

                    // copy data back to host
                    plssvm::soa_matrix<plssvm::real_type> data_ret{ data };
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

TYPED_TEST_P(GenericCSVMSolverKernelFunction, assemble_kernel_matrix) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming) {
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
                                                      plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    [[maybe_unused]] const std::vector<plssvm::real_type> q_red = { plssvm::real_type{ 3.0 }, plssvm::real_type{ 4.0 } };
    [[maybe_unused]] const plssvm::real_type QA_cost{ 2.0 };

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::assemble_kernel_matrix is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(params, csvm_test_type::additional_arguments);
    const std::size_t num_devices = svm.num_available_devices();
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_rows() - 1, 1);

    // automatic solver type not permitted
    if constexpr (solver == plssvm::solver_type::automatic) {
#if defined(PLSSVM_ASSERT_ENABLED)
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
                if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp) {
                    kernel_matrix = plssvm::detail::move_only_any_cast<std::vector<plssvm::real_type>>(kernel_matrix_d[device_id]);  // std::vector
                    ASSERT_EQ(kernel_matrix.size(), 3);
                } else {
                    const auto &kernel_matrix_d_ptr = plssvm::detail::move_only_any_cast<const device_ptr_type &>(kernel_matrix_d[device_id]);  // device_ptr -> convert it to a std::vector
                    kernel_matrix.resize(kernel_matrix_d_ptr.size_padded());
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
                if constexpr (plssvm::csvm_to_backend_type_v<csvm_type> == plssvm::backend_type::openmp) {
                    const auto &[data_d_ret, params_ret, q_red_ret, QA_cost_ret] = plssvm::detail::move_only_any_cast<const std::tuple<plssvm::aos_matrix<plssvm::real_type>, plssvm::parameter, std::vector<plssvm::real_type>, plssvm::real_type> &>(kernel_matrix_d[device_id]);

                    // the values should not have changed! (except the matrix layout)
                    EXPECT_EQ(params_ret, params);
                    EXPECT_FLOATING_POINT_MATRIX_EQ(data_d_ret, plssvm::aos_matrix<plssvm::real_type>{ data });
                    EXPECT_EQ(q_red_ret, q_red);
                    EXPECT_EQ(QA_cost_ret, QA_cost);
                } else {
                    const auto &[data_d_ret, params_ret, q_red_d_ret, QA_cost_ret] = plssvm::detail::move_only_any_cast<const std::tuple<device_ptr_type, plssvm::parameter, device_ptr_type, plssvm::real_type> &>(kernel_matrix_d[device_id]);

                    // copy data back to host
                    plssvm::soa_matrix<plssvm::real_type> data_ret{ data };
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
                            solve_lssvm_system_of_linear_equations_trivial,
                            solve_lssvm_system_of_linear_equations,
                            solve_lssvm_system_of_linear_equations_with_correction,
                            assemble_kernel_matrix_minimal,
                            assemble_kernel_matrix);

//*************************************************************************************************************************************//
//                                 CSVM tests depending on the kernel function and classification type                                 //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMKernelFunctionClassification : public GenericCSVM<T> { };

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

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMKernelFunctionClassification,
                            predict,
                            score_model,
                            score);

//*************************************************************************************************************************************//
//                             CSVM tests depending on the solver, kernel function, and classification type                            //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMSolverKernelFunctionClassification : public GenericCSVM<T> { };

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
    if constexpr (solver == plssvm::solver_type::cg_streaming) {
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
    // EXPECT_FLOATING_POINT_VECTOR_NEAR_EPS(model.rho(), correct_model.rho(), plssvm::real_type{ 1e12 });
    EXPECT_EQ(model.get_classification_type(), classification);
    EXPECT_TRUE(model.num_iters().has_value());
    EXPECT_EQ(model.num_iters().value().size(), (plssvm::calculate_number_of_classifiers(classification, correct_model.num_classes())));
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunctionClassification,
                            fit);

//*************************************************************************************************************************************//
//                                                           CSVM DeathTests                                                           //
//*************************************************************************************************************************************//

template <typename T>
class GenericCSVMDeathTest : public GenericCSVM<T> { };

TYPED_TEST_SUITE_P(GenericCSVMDeathTest);

TYPED_TEST_P(GenericCSVMDeathTest, blas_level_3_automatic) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using csvm_type = typename csvm_test_type::csvm_type;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;

    constexpr plssvm::solver_type solver = plssvm::solver_type::automatic;

    // create C-SVM: must be done using the mock class, since plssvm::detail::csvm::blas_level_3 is protected
    const mock_csvm_type svm = util::construct_from_tuple<mock_csvm_type>(csvm_test_type::additional_arguments);

    const plssvm::real_type alpha{ 1.0 };

    const plssvm::aos_matrix<plssvm::real_type> matr_A{
        { { plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.3 } },
          { plssvm::real_type{ 0.2 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 1.3 } },
          { plssvm::real_type{ 0.3 }, plssvm::real_type{ 1.3 }, plssvm::real_type{ 2.3 } } }
    };
    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_explicit_matrices<csvm_type, device_ptr_type>(matr_A, svm) };

    const plssvm::soa_matrix<plssvm::real_type> B{ { { plssvm::real_type{ 1.0 }, plssvm::real_type{ 2.0 }, plssvm::real_type{ 3.0 } },
                                                     { plssvm::real_type{ 4.0 }, plssvm::real_type{ 5.0 }, plssvm::real_type{ 6.0 } },
                                                     { plssvm::real_type{ 7.0 }, plssvm::real_type{ 8.0 }, plssvm::real_type{ 9.0 } } },
                                                   plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    [[maybe_unused]] const plssvm::real_type beta{ 0.0 };
    plssvm::soa_matrix<plssvm::real_type> C{ plssvm::shape{ 3, 3 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    // automatic solver type not permitted
    EXPECT_DEATH(svm.blas_level_3(solver, alpha, A, B, beta, C), "An explicit solver type must be provided instead of solver_type::automatic!");
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMDeathTest,
                            blas_level_3_automatic);

template <typename T>
class GenericCSVMSolverDeathTest : public GenericCSVM<T> { };

TYPED_TEST_SUITE_P(GenericCSVMSolverDeathTest);

TYPED_TEST_P(GenericCSVMSolverDeathTest, conjugate_gradients_empty_B) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm_type svm{};

    const plssvm::soa_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> empty_matr{};

    EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, empty_matr, plssvm::real_type{ 0.001 }, 6, solver), "The right-hand sides must not be empty!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, conjugate_gradients_invalid_eps) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type svm{};

    const plssvm::soa_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    const plssvm::soa_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 6 } };

    EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, B, 0.0, 6, solver), "The epsilon value must be greater than 0.0!");
    EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, B, -0.5, 6, solver), "The epsilon value must be greater than 0.0!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, conjugate_gradients_invalid_max_cg_iter) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type svm{};

    const plssvm::soa_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    const plssvm::soa_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 6 } };

    EXPECT_DEATH(std::ignore = svm.conjugate_gradients(A, B, plssvm::real_type{ 0.001 }, 0, solver), "The maximum number of iterations must be greater than 0!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, run_blas_level_3_empty_B) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type svm{};

    const plssvm::soa_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    const plssvm::soa_matrix<plssvm::real_type> empty_matr{};
    plssvm::soa_matrix<plssvm::real_type> C{ plssvm::shape{ 4, 4 } };

    EXPECT_DEATH(std::ignore = svm.run_blas_level_3(solver, plssvm::real_type{ 1.0 }, A, empty_matr, plssvm::real_type{ 1.0 }, C), "The B matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, run_blas_level_3_empty_C) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type svm{};

    const plssvm::soa_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    const plssvm::soa_matrix<plssvm::real_type> B{ plssvm::shape{ 4, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    plssvm::soa_matrix<plssvm::real_type> empty_matr{};

    EXPECT_DEATH(std::ignore = svm.run_blas_level_3(solver, plssvm::real_type{ 1.0 }, A, B, plssvm::real_type{ 1.0 }, empty_matr), "The C matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, blas_level_3_empty_matrices) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type svm{};

    const plssvm::soa_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    plssvm::soa_matrix<plssvm::real_type> matr{ plssvm::shape{ 4, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    plssvm::soa_matrix<plssvm::real_type> empty_matr{};

    EXPECT_DEATH(svm.blas_level_3(solver, plssvm::real_type{ 1.0 }, A, empty_matr, plssvm::real_type{ 1.0 }, matr), "The B matrix must not be empty!");
    EXPECT_DEATH(svm.blas_level_3(solver, plssvm::real_type{ 1.0 }, A, matr, plssvm::real_type{ 1.0 }, empty_matr), "The C matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMSolverDeathTest, blas_level_3_missing_padding) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    using csvm_type = typename csvm_test_type::csvm_type;
    using device_ptr_type = typename csvm_test_type::device_ptr_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;

    // create mock_csvm
    const mock_csvm_type svm{};

    const plssvm::soa_matrix<plssvm::real_type> matr_A{ plssvm::shape{ 4, 4 } };
    // parameter necessary for cg_implicit
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> q_red(matr_A.num_rows() - 1);
    const plssvm::real_type QA_cost{ 1.0 };

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(matr_A.num_rows(), svm.num_available_devices());
    const std::vector<plssvm::detail::move_only_any> A{ util::init_matrices<csvm_type, device_ptr_type>(matr_A, solver, svm, params, q_red, QA_cost) };

    plssvm::soa_matrix<plssvm::real_type> matr_padded{ plssvm::shape{ 4, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    plssvm::soa_matrix<plssvm::real_type> matr{ plssvm::shape{ 4, 4 } };

    EXPECT_DEATH(svm.blas_level_3(solver, plssvm::real_type{ 1.0 }, A, matr, plssvm::real_type{ 1.0 }, matr_padded), "The B matrix must be padded!");
    EXPECT_DEATH(svm.blas_level_3(solver, plssvm::real_type{ 1.0 }, A, matr_padded, plssvm::real_type{ 1.0 }, matr), "The C matrix must be padded!");
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolverDeathTest,
                            conjugate_gradients_empty_B,
                            conjugate_gradients_invalid_eps,
                            conjugate_gradients_invalid_max_cg_iter,
                            run_blas_level_3_empty_B,
                            run_blas_level_3_empty_C,
                            blas_level_3_empty_matrices,
                            blas_level_3_missing_padding);

template <typename T>
class GenericCSVMKernelFunctionDeathTest : public GenericCSVMKernelFunction<T> { };

TYPED_TEST_SUITE_P(GenericCSVMKernelFunctionDeathTest);

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_empty_A) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> empty_matr{};
    const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 4 } };

    EXPECT_DEATH(std::ignore = svm.solve_lssvm_system_of_linear_equations(empty_matr, B, params), "The A matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_A_without_padding) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> A{ plssvm::shape{ 6, 4 } };
    const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 4 } };

    EXPECT_DEATH(std::ignore = svm.solve_lssvm_system_of_linear_equations(A, B, params), "The A matrix must be padded!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_A_wrong_padding_sizes) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> A{ plssvm::shape{ 6, 4 }, plssvm::shape{ 0, 1 } };
    const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 4 } };

    EXPECT_DEATH(std::ignore = svm.solve_lssvm_system_of_linear_equations(A, B, params),
                 ::testing::HasSubstr(fmt::format("The provided matrix must be padded with [{}, {}], but is padded with [0, 1]!", plssvm::PADDING_SIZE, plssvm::PADDING_SIZE)));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_empty_B) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_matr{};
    const plssvm::soa_matrix<plssvm::real_type> A{ plssvm::shape{ 6, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };

    EXPECT_DEATH(std::ignore = svm.solve_lssvm_system_of_linear_equations(A, empty_matr, params), "The B matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, solve_lssvm_system_of_linear_equations_size_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> A{ plssvm::shape{ 6, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE } };
    const plssvm::aos_matrix<plssvm::real_type> B{ plssvm::shape{ 1, 3 } };

    EXPECT_DEATH(std::ignore = svm.solve_lssvm_system_of_linear_equations(A, B, params), ::testing::HasSubstr("The number of data points in A (6) and B (3) must be the same!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, perform_dimensional_reduction_empty_A) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::soa_matrix<plssvm::real_type> empty_matr{};

    EXPECT_DEATH(std::ignore = svm.perform_dimensional_reduction(params, empty_matr), "The matrix must not be empty!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, assemble_kernel_matrix_automatic) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto A = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 5 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> q_red(A.num_rows() - 1);
    const plssvm::real_type QA_cost = 42.0;

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(A.num_rows() - 1, svm.num_available_devices());

    // the solver type must not be automatic
    EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(plssvm::solver_type::automatic, params, A, q_red, QA_cost), ::testing::HasSubstr("An explicit solver type must be provided instead of solver_type::automatic!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_empty_matrices) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_aos_matr{};
    const plssvm::soa_matrix<plssvm::real_type> empty_soa_matr{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> rho(2);
    plssvm::soa_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_rows(), svm.num_available_devices());

    // support vectors shouldn't be empty
    EXPECT_DEATH(std::ignore = svm.predict_values(params, empty_soa_matr, weights, rho, w, data), "The support vectors must not be empty!");
    // weights shouldn't be empty
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, empty_aos_matr, rho, w, data), ::testing::HasSubstr("The alpha vectors (weights) must not be empty!"));
    // predict points shouldn't be empty
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, empty_soa_matr), "The data points to predict must not be empty!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_missing_padding) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create empty matrix
    const plssvm::aos_matrix<plssvm::real_type> empty_aos_matr{};
    const plssvm::soa_matrix<plssvm::real_type> empty_soa_matr{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto support_vectors_without_padding = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto weights_without_padding = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });
    const std::vector<plssvm::real_type> rho(2);
    auto w = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    auto w_without_padding = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto data_without_padding = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_rows(), svm.num_available_devices());

    // support vectors must be padded
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors_without_padding, weights, rho, w, data), "The support vectors must be padded!");
    // weights must be padded
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights_without_padding, rho, w, data), ::testing::HasSubstr("The alpha vectors (weights) must be padded!"));
    // w must be padded
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w_without_padding, data_without_padding), "Either w must be empty or must be padded!");
    // predict points must be padded
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data_without_padding), "The data points to predict must be padded!");
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_sv_alpha_size_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 3, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> rho(2);
    plssvm::soa_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_rows(), svm.num_available_devices());

    // the number of support vectors and weights must be identical
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("The number of support vectors (3) and number of weights (4) must be the same!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_rho_alpha_size_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> rho(1);
    plssvm::soa_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_rows(), svm.num_available_devices());

    // the number of rho values and weight vectors must be identical
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("The number of rho values (1) and the number of weight vectors (2) must be the same!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_w_size_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> rho(2);
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_rows(), svm.num_available_devices());

    // the number of features and w values must be identical
    auto w = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 3 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("Either w must be empty or contain exactly the same number of values (3) as features are present (4)!"));
    // the number of weight and w vectors must be identical
    w = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 3, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("Either w must be empty or contain exactly the same number of vectors (3) as the alpha vector (2)!"));
}

TYPED_TEST_P(GenericCSVMKernelFunctionDeathTest, predict_values_num_features_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<0, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto support_vectors = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 5 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const auto weights = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> rho(2);
    plssvm::soa_matrix<plssvm::real_type> w{};
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 2, 4 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(data.num_rows(), svm.num_available_devices());

    // the number of features for the support vectors and predict points must be identical
    EXPECT_DEATH(std::ignore = svm.predict_values(params, support_vectors, weights, rho, w, data), ::testing::HasSubstr("The number of features in the support vectors (5) must be the same as in the data points to predict (4)!"));
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMKernelFunctionDeathTest,
                            solve_lssvm_system_of_linear_equations_empty_A,
                            solve_lssvm_system_of_linear_equations_A_without_padding,
                            solve_lssvm_system_of_linear_equations_A_wrong_padding_sizes,
                            solve_lssvm_system_of_linear_equations_empty_B,
                            solve_lssvm_system_of_linear_equations_size_mismatch,
                            perform_dimensional_reduction_empty_A,
                            assemble_kernel_matrix_automatic,
                            predict_values_empty_matrices,
                            predict_values_missing_padding,
                            predict_values_sv_alpha_size_mismatch,
                            predict_values_rho_alpha_size_mismatch,
                            predict_values_w_size_mismatch,
                            predict_values_num_features_mismatch);

template <typename T>
class GenericCSVMSolverKernelFunctionDeathTest : public GenericCSVMSolverKernelFunction<T> { };

TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunctionDeathTest);

TYPED_TEST_P(GenericCSVMSolverKernelFunctionDeathTest, assemble_kernel_matrix_empty_matrices) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto A = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 5 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> q_red(A.num_rows() - 1);
    const plssvm::real_type QA_cost = 42.0;

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(A.num_rows() - 1, svm.num_available_devices());

    const plssvm::soa_matrix<plssvm::real_type> empty_matr{};
    const std::vector<plssvm::real_type> empty_vec{};

    // the A matrix must not be empty
    EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(solver, params, empty_matr, q_red, QA_cost), ::testing::HasSubstr("The matrix to setup on the devices must not be empty!"));
    // the q_red vector must not be empty
    EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(solver, params, A, empty_vec, QA_cost), ::testing::HasSubstr("The q_red vector must not be empty!"));
}

TYPED_TEST_P(GenericCSVMSolverKernelFunctionDeathTest, assemble_kernel_matrix_A_not_padded) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto A = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 5 });
    const std::vector<plssvm::real_type> q_red(A.num_rows() - 1);
    const plssvm::real_type QA_cost = 42.0;

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(A.num_rows() - 1, svm.num_available_devices());

    // the A matrix must be padded
    EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(solver, params, A, q_red, QA_cost), ::testing::HasSubstr("Tha matrix to setup on the devices must be padded!"));
}

TYPED_TEST_P(GenericCSVMSolverKernelFunctionDeathTest, assemble_kernel_matrix_size_mismatch) {
    using csvm_test_type = util::test_parameter_type_at_t<0, TypeParam>;
    using mock_csvm_type = typename csvm_test_type::mock_csvm_type;
    constexpr plssvm::solver_type solver = util::test_parameter_value_at_v<0, TypeParam>;
    constexpr plssvm::kernel_function_type kernel = util::test_parameter_value_at_v<1, TypeParam>;

    // create parameter
    const plssvm::parameter params{ plssvm::kernel_type = kernel };

    // create mock_csvm
    const mock_csvm_type svm{};

    // create correct input matrices
    const auto A = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 5 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
    const std::vector<plssvm::real_type> q_red(A.num_rows());
    const plssvm::real_type QA_cost = 42.0;

    // be sure to use the correct data distribution
    svm.data_distribution_ = std::make_unique<plssvm::detail::triangular_data_distribution>(A.num_rows() - 1, svm.num_available_devices());

    // the A matrix must be padded
    EXPECT_DEATH(std::ignore = svm.assemble_kernel_matrix(solver, params, A, q_red, QA_cost), ::testing::HasSubstr("The q_red size (4) mismatches the number of data points after dimensional reduction (3)!"));
}

REGISTER_TYPED_TEST_SUITE_P(GenericCSVMSolverKernelFunctionDeathTest,
                            assemble_kernel_matrix_empty_matrices,
                            assemble_kernel_matrix_A_not_padded,
                            assemble_kernel_matrix_size_mismatch);

#endif  // PLSSVM_TESTS_BACKENDS_GENERIC_CSVM_TESTS_HPP_
