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
#if defined(PLSSVM_THREAD_BLOCK_SIZE)
constexpr unsigned int THREAD_BLOCK_SIZE = PLSSVM_THREAD_BLOCK_SIZE;
#else
constexpr unsigned int THREAD_BLOCK_SIZE = 16;
#endif

/// Used for internal caching.
#if defined(PLSSVM_INTERNAL_BLOCK_SIZE)
constexpr unsigned int INTERNAL_BLOCK_SIZE = PLSSVM_INTERNAL_BLOCK_SIZE;
#else
constexpr unsigned int INTERNAL_BLOCK_SIZE = 6;
#endif

static_assert(THREAD_BLOCK_SIZE > 0, "THREAD_BLOCK_SIZE must be greater than 0!");
static_assert(INTERNAL_BLOCK_SIZE > 0, "INTERNAL_BLOCK_SIZE must be greater than 0!");

}  // namespace plssvm