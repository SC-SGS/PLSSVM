/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM class using the OpenMP backend.
 */

#ifndef PLSSVM_TESTS_BACKENDS_OPENMP_MOCK_OPENMP_CSVM_HPP_
#define PLSSVM_TESTS_BACKENDS_OPENMP_MOCK_OPENMP_CSVM_HPP_
#pragma once

#include "plssvm/backends/OpenMP/csvm.hpp"   // plssvm::openmp::csvm
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"              // plssvm::parameter
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

#include <vector>  // std::vector

/**
 * @brief GTest mock class for the OpenMP CSVM.
 */
class mock_openmp_csvm final : public plssvm::openmp::csvm {
    using base_type = plssvm::openmp::csvm;

  public:
    using size_type = typename base_type::size_type;

    explicit mock_openmp_csvm(const plssvm::target_platform target, const plssvm::parameter params = {}) :
        base_type{ target, params } {}
    template <typename... Args>
    mock_openmp_csvm(const plssvm::target_platform target, const plssvm::kernel_function_type kernel, Args &&...named_args) :
        base_type{ target, kernel, std::forward<Args>(named_args)... } {}
};

#endif  // PLSSVM_TESTS_BACKENDS_OPENMP_MOCK_OPENMP_CSVM_HPP_