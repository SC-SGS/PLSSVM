/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*/

#include "plssvm/detail/execution_range.hpp"

#include "plssvm/detail/assert.hpp"  // PLSSVM_ASSERT

#include "fmt/format.h"  // fmt::format, fmt::join

#include <algorithm>         // std::copy
#include <array>             // std::array
#include <cstddef>           // std::size_t
#include <initializer_list>  // std::initializer_list
#include <ostream>           // std::ostream

namespace plssvm::detail {

execution_range::execution_range(const std::initializer_list<std::size_t> p_grid, const std::initializer_list<std::size_t> p_block) {
    PLSSVM_ASSERT(0 < p_grid.size() && p_grid.size() <= 3, fmt::format("The number of grid sizes specified must be between 1 and 3, but is {}!", p_grid.size()));
    PLSSVM_ASSERT(0 < p_block.size() && p_block.size() <= 3, fmt::format("The number of block sizes specified must be between 1 and 3, but is {}!", p_block.size()));

    std::copy(p_grid.begin(), p_grid.end(), grid.begin());
    std::copy(p_block.begin(), p_block.end(), block.begin());
}

std::ostream &operator<<(std::ostream &out, const execution_range &range) {
    return out << fmt::format("grid: [{}]; block: [{}]", fmt::join(range.grid, " "), fmt::join(range.block, " "));
}

}  // namespace plssvm::detail