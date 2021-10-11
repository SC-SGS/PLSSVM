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

#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::exception

#include <algorithm>         // std::copy
#include <array>             // std::array
#include <initializer_list>  // std::initializer_list
#include <ostream>           // std::ostream
#include <type_traits>       // std::enable_if_t

#include "fmt/format.h"  // fmt::format, fmt::join

namespace plssvm::detail {

/**
 * @brief Class specifying the backend independent execution range.
 * @details Holds two members: `grid` specifying the grid size and `block` specifying the block size using the CUDA definition.
 *          Both grid and block must specify at least one and at most three values used in the kernel invocation.
 * @tparam T the type of the grid and block specifying an execution_range
 */
template <typename T>
class execution_range {
  public:
    /**
     * @brief Initialize the grid and block sizes using [`std::initializer_list`](https://en.cppreference.com/w/cpp/utility/initializer_list)s.
     * @details If less than three values are specified, fills the missing values with zero.
     *          Uses the CUDA definition.
     * @throws `plssvm::exception` if the number of values specified for the grid and block sizes are less than one or greater than three
     * @param[in] p_grid specifies the grid sizes
     * @param[in] p_block specifies the block sizes
     */
    execution_range(const std::initializer_list<T> p_grid, const std::initializer_list<T> p_block) {
        if (p_grid.size() <= 0 || p_grid.size() > 3) {
            throw exception{ fmt::format("The number of grid sizes specified must be between 1 and 3, but is {}!", p_grid.size()) };
        } else if (p_block.size() <= 0 || p_block.size() > 3) {
            throw exception{ fmt::format("The number of block sizes specified must be between 1 and 3, but is {}!", p_block.size()) };
        }

        std::copy(p_grid.begin(), p_grid.end(), grid.begin());
        std::copy(p_block.begin(), p_block.end(), block.begin());
    }

    /**
     * @brief Initialize the grid and block sizes using [`std::array`](https://en.cppreference.com/w/cpp/container/array)s.
     *        Only available if the number of values specified for the grid and block sizes are greater than zero and less or equal to three.
     * @details If less than three values are specified, fills the missing values with zero.
     *          Uses the CUDA definition.
     * @param[in] p_grid specifies the grid sizes
     * @param[in] p_block specifies the block sizes
     */
    template <std::size_t I, std::size_t J, std::enable_if_t<(0 < I && I <= 3 && 0 < J && J <= 3), bool> = true>
    execution_range(const std::array<T, I> &p_grid, const std::array<T, J> &p_block) {
        std::copy(p_grid.begin(), p_grid.end(), grid.begin());
        std::copy(p_block.begin(), p_block.end(), block.begin());
    }

    /**
     * @brief Stream-insertion operator overload for convenient printing of execution range encapsulated by @p range.
     * @param[in,out] out the output-stream to write the kernel type to
     * @param[in] range the execution range
     * @return the output-stream
     */
    friend std::ostream &operator<<(std::ostream &out, const execution_range &range) {
        return out << fmt::format("grid: [{}]; block: [{}]", fmt::join(range.grid, " "), fmt::join(range.block, " "));
    }

    /// The grid sizes.
    std::array<T, 3> grid = { 1, 1, 1 };
    /// The block sizes.
    std::array<T, 3> block = { 1, 1, 1 };
};

}  // namespace plssvm::detail