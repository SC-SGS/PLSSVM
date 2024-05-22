/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a struct containing the GPU backend's execution ranges.
 */

#ifndef PLSSVM_BACKENDS_EXECUTION_RANGE_HPP_
#define PLSSVM_BACKENDS_EXECUTION_RANGE_HPP_

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::kernel_launch_resources

#include "fmt/core.h"  // fmt::format

#include <algorithm>  // std::min
#include <cmath>      // std::ceil
#include <utility>    // std::move, std::pair
#include <vector>     // std::vector

namespace plssvm::detail {

/**
 * @brief A type encapsulating up-to three dimensions for kernel launches.
 */
struct dim_type {
    /**
     * @brief Construct an empty dimensional type.
     */
    dim_type() = default;

    /**
     * @brief Construct an one-dimensional dimensional type.
     */
    explicit dim_type(const unsigned long long x_p) :
        x{ x_p } { }

    /**
     * @brief Construct a two-dimensional dimensional type.
     */
    dim_type(const unsigned long long x_p, const unsigned long long y_p) :
        x{ x_p },
        y{ y_p } { }

    /**
     * @brief Construct a three-dimensional dimensional type.
     */
    dim_type(const unsigned long long x_p, const unsigned long long y_p, const unsigned long long z_p) :
        x{ x_p },
        y{ y_p },
        z{ z_p } { }

    /// The dimensional size in x direction.
    unsigned long long x{ 1 };
    /// The dimensional size in y direction.
    unsigned long long y{ 1 };
    /// The dimensional size in z direction.
    unsigned long long z{ 1 };
};

/**
 * @brief A struct encapsulating an arbitrary execution range used to launch a kernel.
 */
struct execution_range {
    /// The type used to store the grid sizes and offsets.
    using grid_type = std::pair<dim_type, dim_type>;

    /**
     * @brief Create a block and grid(s) used to launch the kernels.
     * @details If the provided grid would be too large to be launched, it is split into multiple sub-grids.
     * @param[in] block_p the requested block size
     * @param[in] max_allowed_block_size the maximum allowed 1D block size (i.e., `block.x * block.y * block.z`)
     * @param[in] grid the requested grid size
     * @param[in] max_allowed_grid_size the maximum allowed 3D grid sizes
     * @throws plssvm::kernel_launch_resources if the block size exceeds the upper limits
     */
    execution_range(const dim_type block_p, const unsigned long long max_allowed_block_size, const dim_type grid, const dim_type max_allowed_grid_size) :
        block{ block_p } {
        // check whether the provided block size is valid
        if (max_allowed_block_size < block.x * block.y * block.z) {
            throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}x{}, max {}! Try reducing THREAD_BLOCK_SIZE.", block.x, block.y, block.z, max_allowed_block_size) };
        }

        // TODO: implement better?!
        // split the large grid into sub-grids
        const unsigned long long num_grid_x = grid.x / max_allowed_grid_size.x;
        for (unsigned long long x = 0; x < num_grid_x; ++x) {
            const unsigned long long num_grid_y = grid.y / max_allowed_grid_size.y;
            for (unsigned long long y = 0; y < num_grid_y; ++y) {
                grids.emplace_back(dim_type{ std::min(grid.x, max_allowed_grid_size.x), std::min(grid.y, max_allowed_grid_size.y) }, dim_type{ x * max_allowed_grid_size.x, y * max_allowed_grid_size.y });
            }
            const unsigned long long remaining_y = grid.y % max_allowed_grid_size.y;
            if (remaining_y > 0ull) {
                grids.emplace_back(dim_type{ std::min(grid.x, max_allowed_grid_size.x), remaining_y }, dim_type{ x * max_allowed_grid_size.x, num_grid_y * max_allowed_grid_size.y });
            }
        }
        const unsigned long long remaining_x = grid.x % max_allowed_grid_size.x;
        if (remaining_x > 0ull) {
            const unsigned long long num_grid_y = grid.y / max_allowed_grid_size.y;
            for (unsigned long long y = 0; y < num_grid_y; ++y) {
                grids.emplace_back(dim_type{ std::min(grid.x, max_allowed_grid_size.x), std::min(grid.y, max_allowed_grid_size.y) }, dim_type{ num_grid_x * max_allowed_grid_size.x, y * max_allowed_grid_size.y });
            }
            const unsigned long long remaining_y = grid.y % max_allowed_grid_size.y;
            if (remaining_y > 0ull) {
                grids.emplace_back(dim_type{ remaining_x, remaining_y }, dim_type{ num_grid_x * max_allowed_grid_size.x, num_grid_y * max_allowed_grid_size.y });
            }
        }
    }

    /// The up-to three dimensional block (work-group) size.
    dim_type block{};
    /// The grids. Multiple grids are used, if the grid sizes would exceed the maximum allowed number. Also stores the offsets for the respective grids used in the kernels.
    std::vector<grid_type> grids{};
};

}  // namespace plssvm::detail

#endif  // PLSSVM_BACKENDS_EXECUTION_RANGE_HPP_
