/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/mpi/environment.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::mpi_exception
#include "plssvm/mpi/detail/utility.hpp"     // PLSSVM_MPI_ERROR_CHECK

#include "fmt/format.h"  // fmt::format

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"
#endif

#include <string>  // std::string

namespace plssvm::mpi {

void init() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    const int required = MPI_THREAD_FUNNELED;
    int provided{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Init_thread(nullptr, nullptr, required, &provided));
    if (required < provided) {
        throw mpi_exception{ fmt::format("Error: provided thread level {} to small for requested thread level {}!", provided, required) };
    }
#endif
}

void init(int &argc, char **argv) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    const int required = MPI_THREAD_FUNNELED;
    int provided{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Init_thread(&argc, &argv, required, &provided));
    if (required < provided) {
        throw mpi_exception{ fmt::format("Error: provided thread level {} to small for requested thread level {}!", provided, required) };
    }
#endif
}

void finalize() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    PLSSVM_MPI_ERROR_CHECK(MPI_Finalize());
#endif
}

bool is_initialized() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    int flag{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Initialized(&flag));
    return static_cast<bool>(flag);
#else
    return true;
#endif
}

bool is_finalized() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    int flag{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Finalized(&flag));
    return static_cast<bool>(flag);
#else
    return true;
#endif
}

}  // namespace plssvm::mpi
