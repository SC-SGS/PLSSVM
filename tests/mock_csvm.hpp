/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM base class.
 */

#ifndef PLSSVM_TESTS_MOCK_CSVM_HPP_
#define PLSSVM_TESTS_MOCK_CSVM_HPP_
#pragma once

#include "plssvm/constants.hpp"           // plssvm::real_type
#include "plssvm/csvm.hpp"                // plssvm::csvm
#include "plssvm/detail/memory_size.hpp"  // plssvm::detail::memory_size, plssvm::detail::literals
#include "plssvm/detail/simple_any.hpp"   // plssvm::detail::simple_any
#include "plssvm/matrix.hpp"              // plssvm::aos_matrix
#include "plssvm/parameter.hpp"           // plssvm::parameter
#include "plssvm/solver_types.hpp"        // plssvm::solver_type

#include "gmock/gmock.h"  // MOCK_METHOD, ON_CALL, ::testing::Return

#include <utility>  // std::forward
#include <vector>   // std::vector

/**
 * @brief GTest mock class for the base CSVM class.
 */
class mock_csvm final : public plssvm::csvm {
  public:
    template <typename... Args>
    explicit mock_csvm(Args &&...args) :
        plssvm::csvm{ std::forward<Args>(args)... } {
        this->fake_functions();
    }

    // mock pure virtual functions
    MOCK_METHOD((plssvm::detail::memory_size), get_device_memory, (), (const, override));
    MOCK_METHOD((plssvm::detail::memory_size), get_max_mem_alloc_size, (), (const, override));
    MOCK_METHOD((plssvm::detail::simple_any), setup_data_on_devices, (plssvm::solver_type, const plssvm::soa_matrix<plssvm::real_type> &), (const, override));
    MOCK_METHOD((plssvm::detail::simple_any), assemble_kernel_matrix, (plssvm::solver_type, const plssvm::parameter &, const plssvm::detail::simple_any &, const std::vector<plssvm::real_type> &, plssvm::real_type), (const, override));
    MOCK_METHOD((void), blas_level_3, (plssvm::solver_type, plssvm::real_type, const plssvm::detail::simple_any &, const plssvm::soa_matrix<plssvm::real_type> &, plssvm::real_type, plssvm::soa_matrix<plssvm::real_type> &), (const, override));
    MOCK_METHOD((plssvm::aos_matrix<plssvm::real_type>), predict_values, (const plssvm::parameter &, const plssvm::soa_matrix<plssvm::real_type> &, const plssvm::aos_matrix<plssvm::real_type> &, const std::vector<plssvm::real_type> &, plssvm::soa_matrix<plssvm::real_type> &, const plssvm::soa_matrix<plssvm::real_type> &), (const, override));

  private:
    void fake_functions() const {
        using namespace plssvm::detail::literals;
        // clang-format off
        ON_CALL(*this, get_device_memory()).WillByDefault(::testing::Return(1_GiB));
        ON_CALL(*this, get_max_mem_alloc_size()).WillByDefault(::testing::Return(512_MiB));
        // clang-format on
    }
};

#endif  // PLSSVM_TESTS_MOCK_CSVM_HPP_