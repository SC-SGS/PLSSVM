/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVR base class.
 */

#ifndef PLSSVM_TESTS_MOCK_CSVR_HPP_
#define PLSSVM_TESTS_MOCK_CSVR_HPP_
#pragma once

#include "plssvm/svm/csvm.hpp"  // plssvm::csvm
#include "plssvm/svm/csvr.hpp"  // plssvm::csvr

#include "tests/svm/mock_csvm.hpp"  // mock_csvm

/**
 * @brief GTest mock class for the base CSVR class.
 */
class mock_csvr final : virtual public plssvm::csvr,
                        public mock_csvm {
  public:
    template <typename... Args>
    explicit mock_csvr(Args... args) :
        plssvm::csvm{ args... },
        mock_csvm{ args... } { }
};

#endif  // PLSSVM_TESTS_MOCK_CSVR_HPP_
