/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @author Alexander Strack
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/environment.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::environment_exception
#include "plssvm/detail/utility.hpp"         // plssvm::detail::{contains, unreachable}
#include "plssvm/mpi/environment.hpp"        // plssvm::mpi::{is_finalized, finalize}

#if defined(PLSSVM_HAS_HPX_BACKEND)
    #include "hpx/execution.hpp"  // ::hpx::post
    #include "hpx/hpx_start.hpp"  // ::hpx::{start, stop, finalize}
    #include "hpx/runtime.hpp"    // ::hpx::{is_running, is_stopped}
#endif
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    #include "Kokkos_Core.hpp"  // Kokkos::is_initialized, Kokkos::is_finalized, Kokkos::initialize, Kokkos::finalize
#endif

#include "fmt/format.h"   // fmt::format
#include "fmt/ranges.h"   // fmt::join

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <vector>     // std::vector
#include <algorithm>  // std::remove_if

namespace plssvm::environment {

std::ostream &operator<<(std::ostream &out, const status s) {
    switch (s) {
        case status::uninitialized:
            return out << "uninitialized";
        case status::initialized:
            return out << "initialized";
        case status::finalized:
            return out << "finalized";
        case status::unnecessary:
            return out << "unnecessary";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, status &s) {
    std::string str;
    in >> str;
    ::plssvm::detail::to_lower_case(str);

    if (str == "uninitialized") {
        s = status::uninitialized;
    } else if (str == "initialized") {
        s = status::initialized;
    } else if (str == "finalized") {
        s = status::finalized;
    } else if (str == "unnecessary") {
        s = status::unnecessary;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

namespace detail {

status determine_status_from_initialized_finalized_flags(const bool is_initialized, const bool is_finalized) {
    if (!is_initialized) {
        // Note: ::hpx::is_stopped does return true even before calling finalize once
        return status::uninitialized;
    } else if (is_initialized && !is_finalized) {
        return status::initialized;
    } else if (is_finalized) {
        return status::finalized;
    }
    // should never be reached!
    ::plssvm::detail::unreachable();
}

}  // namespace detail

status get_backend_status(const backend_type backend) {
    // Note: must be implemented for the backends that need environmental setup
    switch (backend) {
        case backend_type::automatic:
            // it is unsupported to get the environment status for the automatic backend
            throw environment_exception{ "Can't retrieve the environment status for the automatic backend!" };
        case backend_type::openmp:
        case backend_type::stdpar:
        case backend_type::cuda:
        case backend_type::hip:
        case backend_type::opencl:
        case backend_type::sycl:
            // no environment necessary to manage these backends
            return status::unnecessary;
        case backend_type::hpx:
            {
#if defined(PLSSVM_HAS_HPX_BACKEND)
                return detail::determine_status_from_initialized_finalized_functions<::hpx::is_running, ::hpx::is_stopped>();
#else
                return status::unnecessary;
#endif
            }
        case backend_type::kokkos:
            {
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
                return detail::determine_status_from_initialized_finalized_functions<Kokkos::is_initialized, Kokkos::is_finalized>();
#else
                return status::unnecessary;
#endif
            }
    }
    // should never be reached!
    ::plssvm::detail::unreachable();
}

namespace detail {

void initialize_backend([[maybe_unused]] const backend_type backend) {
    PLSSVM_ASSERT(backend != backend_type::automatic, "The automatic backend may never be initialized!");
    // Note: must be implemented for the backends that need environmental setup
    // only have to perform special initialization steps for the HPX backend
#if defined(PLSSVM_HAS_HPX_BACKEND)
    if (backend == backend_type::hpx) {
        ::hpx::start(nullptr, 0, nullptr);
    }
#endif
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    if (backend == backend_type::kokkos) {
        Kokkos::initialize();
    }
#endif
}

void initialize_backend([[maybe_unused]] const backend_type backend, [[maybe_unused]] int &argc, [[maybe_unused]] char **argv) {
    PLSSVM_ASSERT(backend != backend_type::automatic, "The automatic backend may never be initialized!");
    // Note: must be implemented for the backends that need environmental setup
    // only have to perform special initialization steps for the HPX backend
#if defined(PLSSVM_HAS_HPX_BACKEND)
    if (backend == backend_type::hpx) {
        ::hpx::start(nullptr, argc, argv);
    }
#endif
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    if (backend == backend_type::kokkos) {
        Kokkos::initialize(argc, argv);
    }
#endif
}

void finalize_backend([[maybe_unused]] const backend_type backend) {
    PLSSVM_ASSERT(backend != backend_type::automatic, "The automatic backend may never be finalized!");
    // Note: must be implemented for the backends that need environmental setup
    // only have to perform special initialization steps for the HPX backend
#if defined(PLSSVM_HAS_HPX_BACKEND)
    if (backend == backend_type::hpx) {
        ::hpx::post([] { ::hpx::finalize(); });
        ::hpx::stop();
    }
#endif
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    if (backend == backend_type::kokkos) {
        Kokkos::finalize();
    }
#endif
}

void get_filtered_backends(std::vector<backend_type> &backends, const status s) {
    backends.erase(std::remove_if(backends.begin(), backends.end(), [&s](const backend_type backend) {
                       if (backend == backend_type::automatic) {
                           // always remove the automatic backend
                           return true;
                       } else {
                           // remove all backends for which the filter isn't true
                           return get_backend_status(backend) != s;
                       }
                   }),
                   backends.end());
}

}  // namespace detail

void initialize(const std::vector<backend_type> &backends) {
    detail::initialize_impl(backends);
}

std::vector<backend_type> initialize() {
    // get all available backends
    std::vector<backend_type> backends = list_available_backends();
    // only initialize currently uninitialized backends; remove the automatic backend
    detail::get_filtered_backends(backends, status::uninitialized);
    // initialize the remaining backends
    detail::initialize_impl(backends);
    return backends;
}

void initialize(int &argc, char **argv, const std::vector<backend_type> &backends) {
    detail::initialize_impl(backends, argc, argv);
}

std::vector<backend_type> initialize(int &argc, char **argv) {
    // get all available backends
    std::vector<backend_type> backends = list_available_backends();
    // only initialize currently uninitialized backends; remove the automatic backend
    detail::get_filtered_backends(backends, status::uninitialized);
    // initialize the remaining backends
    detail::initialize_impl(backends, argc, argv);
    return backends;
}

void finalize(const std::vector<backend_type> &backends) {
    // if necessary, finalize MPI
    if (!mpi::is_finalized()) {
        mpi::finalize();
    }

    // check if the provided backends are currently available
    const std::vector<backend_type> available_backends = list_available_backends();
    for (const backend_type backend : backends) {
        // check if the backend is available
        if (!::plssvm::detail::contains(available_backends, backend)) {
            throw environment_exception{ fmt::format("The provided backend {} is currently not available and, therefore, can't be finalized! Available backends are: [{}].", backend, fmt::join(available_backends, ", ")) };
        }
    }

    for (const backend_type backend : backends) {
        // the automatic backend cannot be finalized
        if (backend == backend_type::automatic) {
            throw environment_exception{ "The automatic backend cannot be finalized!" };
        }

        // check the status of the current backend
        switch (get_backend_status(backend)) {
            case status::uninitialized:
                // currently uninitialized -> throw exception
                throw environment_exception{ fmt::format("The backend {} has not been initialized yet!", backend) };
            case status::initialized:
                // backend initialized -> finalize
                detail::finalize_backend(backend);
                break;
            case status::finalized:
                // backend already finalized -> throw exception
                throw environment_exception{ fmt::format("The backend {} has already been finalized!", backend) };
            case status::unnecessary:
                // no initialization or finalization necessary -> do nothing
                break;
        }
    }
}

std::vector<backend_type> finalize() {
    // get all available backends
    std::vector<backend_type> backends = list_available_backends();
    // only finalize currently initialized backends; remove the automatic backend
    detail::get_filtered_backends(backends, status::initialized);
    // finalize the remaining backends
    finalize(backends);
    return backends;
}

}  // namespace plssvm::environment
