/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/mpi/detail/version.hpp"

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "plssvm/mpi/detail/utility.hpp"  // PLSSVM_MPI_ERROR_CHECK

    #include "fmt/format.h"  // fmt::format
    #include "mpi.h"         // MPI_Get_library_version, MPI_Get_version
#endif

#include <string>  // std::string

namespace plssvm::mpi::detail {

std::string mpi_library_version() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    std::string version(MPI_MAX_LIBRARY_VERSION_STRING, '\0');
    int resultlen{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Get_library_version(version.data(), &resultlen));
    return version.substr(0, version.find_first_of('\0'));
#else
    return std::string{ "unknown/unused" };
#endif
}

std::string mpi_version() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    int version{};
    int subversion{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Get_version(&version, &subversion));
    return fmt::format("{}.{}", version, subversion);
#else
    return std::string{ "unknown/unused" };
#endif
}

}  // namespace plssvm::mpi::detail
