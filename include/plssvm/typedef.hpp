/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Global compile-time constants.
 */

#pragma once

namespace plssvm {

/// Used for internal caching.
constexpr unsigned int THREADBLOCK_SIZE = 16;
/// Used for internal caching.
constexpr unsigned int INTERNALBLOCK_SIZE = 6;

// sanity checks TODO: check CUDA limitations
static_assert(THREADBLOCK_SIZE > 0, "THREADBLOCK_SIZE must be greater than 0!");
static_assert(INTERNALBLOCK_SIZE > 0, "INTERNALBLOCK_SIZE must be greater than 0!");

}  // namespace plssvm