/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/mpi/detail/utility.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::mpi_exception

#include "fmt/format.h"  // fmt::format

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"  // MPI_Get_processor_name
#endif

#include <string>  // std::string

namespace plssvm::mpi::detail {

void mpi_error_check([[maybe_unused]] const int err) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    if ((err) != MPI_SUCCESS) {
        std::string err_str(MPI_MAX_ERROR_STRING, '\0');
        int err_str_len{};
        const int res = MPI_Error_string(err, err_str.data(), &err_str_len);
        if (res == MPI_SUCCESS) {
            throw plssvm::mpi_exception{ fmt::format("MPI error {}: {}", err, err_str.substr(0, err_str.find_first_of('\0'))) };
        } else {
            throw plssvm::mpi_exception{ fmt::format("MPI error {}", err) };
        }
    }
#endif
}

std::string node_name() {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    std::string name(MPI_MAX_PROCESSOR_NAME, '\0');
    int resultlen{};
    PLSSVM_MPI_ERROR_CHECK(MPI_Get_processor_name(name.data(), &resultlen));
    return name.substr(0, name.find_first_of('\0'));
#else
    return std::string{ "unknown/unused" };
#endif
}

}  // namespace plssvm::mpi::detail
