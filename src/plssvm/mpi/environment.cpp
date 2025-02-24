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
    #include "mpi.h"  // MPI_THREAD_FUNNELED, MPI_Init_thread, MPI_Finalize, MPI_Initialized, MPI_Finalized
#endif

#include <cstdlib>  // EXIT_FAILURE, std::getenv

namespace plssvm::mpi {

void init() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    constexpr int required = MPI_THREAD_FUNNELED;
    int provided{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Init_thread(nullptr, nullptr, required, &provided));
    if (required < provided) {
        throw mpi_exception{ fmt::format("Error: provided thread level {} to small for requested thread level {}!", provided, required) };
    }
#endif
}

void init([[maybe_unused]] int &argc, [[maybe_unused]] char **argv) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    constexpr int required = MPI_THREAD_FUNNELED;
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

void abort_world() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    PLSSVM_MPI_ERROR_CHECK(MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE));
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

bool is_active() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    return is_initialized() && !is_finalized();
#else
    return false;
#endif
}

bool is_executed_via_mpirun() {
    return std::getenv("OMPI_COMM_WORLD_SIZE") != nullptr ||  // OpenMPI
           std::getenv("PMI_SIZE") != nullptr ||              // MPICH, IntelMPI, OpenMPI
           std::getenv("SLURM_PROCID") != nullptr;            // SLURM
}

}  // namespace plssvm::mpi
