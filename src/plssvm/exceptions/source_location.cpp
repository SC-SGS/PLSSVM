/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/exceptions/source_location.hpp"

#include "plssvm/mpi/communicator.hpp"  // plssvm::mpi::communicator
#include "plssvm/mpi/environment.hpp"   // plssvm::mpi::is_active

#include <cstdint>   // std::uint_least32_t
#include <optional>  // std::make_optional

namespace plssvm {

source_location source_location::current(const char *file_name, const char *function_name, int line, int column) noexcept {
    source_location loc;

    loc.file_name_ = file_name;
    loc.function_name_ = function_name;
    loc.line_ = static_cast<std::uint_least32_t>(line);
    loc.column_ = static_cast<std::uint_least32_t>(column);

    // try getting the MPI rank wrt to MPI_COMM_WORLD
    try {
        if (mpi::is_active()) {
            loc.world_rank_ = std::make_optional(mpi::communicator{}.rank());
        }
    } catch (...) {
        // std::nullopt
    }

    return loc;
}

}  // namespace plssvm
