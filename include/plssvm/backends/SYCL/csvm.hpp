/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines a C-SVM using the SYCL backend.
 */

#pragma once

#include "plssvm/csvm.hpp"  // plssvm::csvm

namespace plssvm::sycl {

/**
 * @brief The C-SVM class using the SYCL backend.
 * @tparam T the type of the data
 */
template <typename T>
class csvm : ::plssvm::csvm<T> {
};

}  // namespace plssvm::sycl