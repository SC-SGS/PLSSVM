/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/mpi/wrapper.hpp"

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::mpi_exception

#include "fmt/format.h"  // fmt::format

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"
#endif

#include <string>  // std::string

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #define PLSSVM_MPI_ERROR_CHECK(err)                                                                                              \
        if ((err) != MPI_SUCCESS) {                                                                                                  \
            std::string err_str(MPI_MAX_ERROR_STRING, '\0');                                                                         \
            int err_str_len{};                                                                                                       \
            const int res = MPI_Error_string(err, err_str.data(), &err_str_len);                                                     \
            if (res == MPI_SUCCESS) {                                                                                                \
                throw plssvm::mpi_exception{ fmt::format("MPI error {}: {}", err, err_str.substr(0, err_str.find_first_of('\0'))) }; \
            } else {                                                                                                                 \
                throw plssvm::mpi_exception{ fmt::format("MPI error {}", err) };                                                     \
            }                                                                                                                        \
        }
#else
    #define PLSSVM_MPI_ERROR_CHECK(...)
#endif

namespace plssvm::detail::mpi {

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

}  // namespace plssvm::detail::mpi
