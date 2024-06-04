/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/execution_range.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::kernel_launch_resources

#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join

#include <algorithm>  // std::min, std::swap
#include <cmath>      // std::ceil
#include <ostream>    // std::ostream
#include <vector>     // std::vector

namespace plssvm::detail {

//*************************************************************************************************************************************//
//                                                              dim_type                                                               //
//*************************************************************************************************************************************//

std::ostream &operator<<(std::ostream &out, const dim_type dim) {
    return out << fmt::format("[{}, {}, {}]", dim.x, dim.y, dim.z);
}

//*************************************************************************************************************************************//
//                                                           execution_range                                                           //
//*************************************************************************************************************************************//

execution_range::execution_range(const dim_type block_p, const unsigned long long max_allowed_block_size, const dim_type grid, const dim_type max_allowed_grid_size) :
    block{ block_p } {
    // check whether the provided block size is valid
    if (this->num_threads_in_block() == 0) {
        throw kernel_launch_resources{ "At least one thread must be given per block! Maybe one dimension is zero?" };
    } else if (max_allowed_block_size < this->num_threads_in_block()) {
        throw kernel_launch_resources{ fmt::format("Not enough work-items allowed for a work-groups of size {}x{}x{} (#threads: {}; max allowed: {})! Try reducing THREAD_BLOCK_SIZE.", block.x, block.y, block.z, this->num_threads_in_block(), max_allowed_block_size) };
    }

    // split the large grid into sub-grids
    const auto num_grid_x = static_cast<unsigned long long>(std::ceil(static_cast<double>(grid.x) / static_cast<double>(max_allowed_grid_size.x)));
    const auto num_grid_y = static_cast<unsigned long long>(std::ceil(static_cast<double>(grid.y) / static_cast<double>(max_allowed_grid_size.y)));
    const auto num_grid_z = static_cast<unsigned long long>(std::ceil(static_cast<double>(grid.z) / static_cast<double>(max_allowed_grid_size.z)));

    unsigned long long remaining_x = grid.x;
    for (unsigned long long x = 0; x < num_grid_x; ++x) {
        unsigned long long remaining_y = grid.y;
        for (unsigned long long y = 0; y < num_grid_y; ++y) {
            unsigned long long remaining_z = grid.z;
            for (unsigned long long z = 0; z < num_grid_z; ++z) {
                grids.emplace_back(dim_type{ std::min(remaining_x, max_allowed_grid_size.x), std::min(remaining_y, max_allowed_grid_size.y), std::min(remaining_z, max_allowed_grid_size.z) },
                                   dim_type{ x * max_allowed_grid_size.x, y * max_allowed_grid_size.y, z * max_allowed_grid_size.z });
                remaining_z -= max_allowed_grid_size.z;
            }
            remaining_y -= max_allowed_grid_size.y;
        }
        remaining_x -= max_allowed_grid_size.x;
    }
}

void execution_range::swap(execution_range &other) noexcept {
    using std::swap;
    swap(block, other.block);
    swap(grids, other.grids);
}

unsigned long long execution_range::num_threads_in_block() const noexcept {
    return block.x * block.y * block.z;
}

void swap(execution_range &lhs, execution_range &rhs) noexcept {
    lhs.swap(rhs);
}

std::ostream &operator<<(std::ostream &out, const execution_range &exec) {
    if (exec.grids.size() == 1) {
        return out << fmt::format("grid: {}; block: {}", exec.grids.front().first, exec.block);
    } else {
        std::vector<dim_type> transformed_vec(exec.grids.size());
        std::transform(exec.grids.cbegin(), exec.grids.cend(), transformed_vec.begin(), [](const execution_range::grid_type &grid) { return grid.first; });
        return out << fmt::format("grids: [{}]; block: {}", fmt::join(transformed_vec, ", "), exec.block);
    }
}

}  // namespace plssvm::detail
