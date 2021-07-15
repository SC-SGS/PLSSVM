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
// TODO: maybe not as global constants?
// TODO: unsigned or std::size_t?
// TODO: docu

/// Used for internal caching.
constexpr unsigned THREADBLOCK_SIZE = 16;
/// Used for internal caching.
constexpr unsigned INTERNALBLOCK_SIZE = 6;
/// Used for internal caching.
constexpr unsigned THREADS_PER_BLOCK = 1024;

// sanity checks TODO: additional checks?
static_assert(THREADBLOCK_SIZE > 0, "THREADBLOCK_SIZE must be greater than 0!");
static_assert(INTERNALBLOCK_SIZE > 0, "INTERNALBLOCK_SIZE must be greater than 0!");
static_assert(THREADS_PER_BLOCK > 0, "THREADS_PER_BLOCK must be greater than 0!");

}  // namespace plssvm