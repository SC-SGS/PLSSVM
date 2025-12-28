/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVC base class.
 */

#ifndef PLSSVM_TESTS_CSVC_MOCK_CSVC_HPP_
#define PLSSVM_TESTS_CSVC_MOCK_CSVC_HPP_
#pragma once

#include "plssvm/mpi/communicator.hpp"  // plssvm::mpi::communicator
#include "plssvm/svm/csvc.hpp"          // plssvm::csvc
#include "plssvm/svm/csvm.hpp"          // plssvm::csvm

#include "tests/svm/mock_csvm.hpp"  // mock_csvm

/**
 * @brief GTest mock class for the base C-SVC class.
 */
class mock_csvc final : virtual public plssvm::csvc,
                        public mock_csvm {
  public:
    template <typename... Args>
    explicit mock_csvc(Args... args) :
        plssvm::csvm{ plssvm::mpi::communicator{}, args... },
        mock_csvm{ args... } { }
};

#endif  // PLSSVM_TESTS_CSVC_MOCK_CSVC_HPP_
