/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implement a backend independent class used to specify the execution range for all kernel invocations.
 */

#ifndef PLSSVM_DETAIL_EXECUTION_RANGE_HPP_
#define PLSSVM_DETAIL_EXECUTION_RANGE_HPP_
#pragma once

#include "plssvm/detail/type_traits.hpp"  // PLSSVM_REQUIRES

#include <algorithm>                      // std::copy
#include <array>                          // std::array
#include <cstddef>                        // std::size_t
#include <initializer_list>               // std::initializer_list
#include <iosfwd>                         // forward declare std::ostream

namespace plssvm::detail {

/**
 * @brief Class specifying a backend independent execution range.
 * @details Holds two members: `grid` specifying the grid size and `block` specifying the block size using the CUDA definition.
 *          Both grid and block must specify at least a one dimensional and at most a three dimensional value used in the kernel invocation.
 */
class execution_range {
  public:
    /**
     * @brief Initialize the grid and block sizes using [`std::initializer_list`](https://en.cppreference.com/w/cpp/utility/initializer_list)s.
     * @details If less than three values are specified, fills the missing values with ones. Uses the CUDA definition.
     * @param[in] grid specifies the grid sizes
     * @param[in] block specifies the block sizes
     */
    execution_range(std::initializer_list<std::size_t> grid, std::initializer_list<std::size_t> block);

    /**
     * @brief Initialize the grid and block sizes using [`std::array`](https://en.cppreference.com/w/cpp/container/array)s.
     *        Only available if the number of values specified for the grid and block sizes are greater than zero and less or equal than three.
     * @details If less than three values are specified, fills the missing values with ones. Uses the CUDA definition.
     * @param[in] p_grid specifies the grid sizes
     * @param[in] p_block specifies the block sizes
     */
    template <std::size_t I, std::size_t J, PLSSVM_REQUIRES((0 < I && I <= 3 && 0 < J && J <= 3))>
    execution_range(const std::array<std::size_t, I> &p_grid, const std::array<std::size_t, J> &p_block) {
        std::copy(p_grid.cbegin(), p_grid.cend(), grid.begin());
        std::copy(p_block.cbegin(), p_block.cend(), block.begin());
    }

    /// The grid sizes (using the CUDA definition).
    std::array<std::size_t, 3> grid = { 1, 1, 1 };
    /// The block sizes (using the CUDA definition).
    std::array<std::size_t, 3> block = { 1, 1, 1 };
};

/**
 * @brief Output the execution @p range to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the execution range to
 * @param[in] range the execution range
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const execution_range &range);

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_EXECUTION_RANGE_HPP_