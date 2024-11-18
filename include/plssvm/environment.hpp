/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Header handling the correct runtime setup and teardown.
 * @note Must be implemented in the header file due to linking of backend specific libraries.
 * @attention This header should **not** be included in any base library file!
 */

#ifndef PLSSVM_ENVIRONMENT_HPP_
#define PLSSVM_ENVIRONMENT_HPP_

#include "plssvm/backend_types.hpp"          // plssvm::backend_type, plssvm::list_available_backends
#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::{contains, unreachable}
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::environment_exception

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter
#include "fmt/ranges.h"   // fmt::join

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <vector>   // std::vector

#if defined(PLSSVM_HAS_HPX_BACKEND)
    #include <hpx/execution.hpp>  // ::hpx::post
    #include <hpx/hpx_start.hpp>  // ::hpx::{start, stop, finalize}
    #include <hpx/runtime.hpp>    // ::hpx::{is_running, is_stopped}
#endif

namespace plssvm::environment {

/**
 * @brief The different possible environment status that our backends can have.
 */
enum class status {
    /** The backend environment hasn't been initialized or finalized yet. */
    uninitialized,
    /** The backend environment has been initialized but not finalized yet. */
    initialized,
    /** The backend environment has already been initialized and finalized. */
    finalized,
    /** No backend environment initialization or finalization necessary. */
    unnecessary
};

/**
 * @brief Output the environment status @p s to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the environment status to
 * @param[in] s the environment status
 * @return the output-stream
 */
inline std::ostream &operator<<(std::ostream &out, const status s) {
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

/**
 * @brief Use the input-stream @p in to initialize the environment status @p s.
 * @param[in,out] in input-stream to extract the environment status from
 * @param[in] s the environment status
 * @return the input-stream
 */
inline std::istream &operator>>(std::istream &in, status &s) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

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

/**
 * @brief Determine the environment status based on the @p is_initialized and @p is_finalized flags.
 * @param is_initialized `true` if the respective environment has been initialized, `false` otherwise
 * @param is_finalized `true` if the respective environment has been finalized, `false` otherwise
 * @return the respective environment status (`[[nodiscard]]`)
 */
[[nodiscard]] inline status determine_status_from_initialized_finalized_flags(const bool is_initialized, const bool is_finalized) {
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

/**
 * @brief Determine the environment status based on the result of the @p is_initialized_function and @p is_finalized_function functions.
 * @tparam is_initialized_function the function to check whether a backend is initialized
 * @tparam is_finalized_function the function to check whether a backend is finalized
 * @return the respective environment status (`[[nodiscard]]`)
 */
template <auto is_initialized_function, auto is_finalized_function>
[[nodiscard]] inline status determine_status_from_initialized_finalized_functions() {
    return determine_status_from_initialized_finalized_flags(is_initialized_function(), is_finalized_function());
}

}  // namespace detail

//****************************************************************************//
//                         environment query function                         //
//****************************************************************************//

/**
 * @brief Get the current status of the environment for the @p backend.
 * @param[in] backend the backend to check the environment status
 * @throws plssvm::environment_exception if @p backend is the automatic backend
 * @return the environment status for the @p backend (`[[nodiscard]]`)
 */
[[nodiscard]] inline status get_backend_status(const backend_type backend) {
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
    }
    // should never be reached!
    ::plssvm::detail::unreachable();
}

/**
 * @brief Check whether the provided backend needs a special initialization.
 * @param[in] backend the backend to check
 * @return `true` if the backend needs a special environment initialization, `false` otherwise
 */
constexpr bool is_initialization_necessary([[maybe_unused]] const backend_type backend) {
    // Note: must be implemented for the backends that need environmental setup
    // currently false for all available backends
    return false;
}

//****************************************************************************//
//         backend specific initialization and finalization functions         //
//****************************************************************************//

namespace detail {

/**
 * @brief Initialize the @p backend.
 * @param[in] backend the backend to initialize
 */
inline void initialize_backend([[maybe_unused]] const backend_type backend) {
    PLSSVM_ASSERT(backend != backend_type::automatic, "The automatic backend may never be initialized!");
    // Note: must be implemented for the backends that need environmental setup
    // only have to perform special initialization steps for the HPX backend
#if defined(PLSSVM_HAS_HPX_BACKEND)
    if (backend == backend_type::hpx) {
        ::hpx::start(nullptr, 0, nullptr);
    }
#endif
}

/**
 * @brief Initialize the @p backend with the provided command line arguments.
 * @param[in] backend the backend to initialize
 * @param[in,out] argc the number of command line arguments
 * @param[in,out] argv the command line arguments
 */
inline void initialize_backend([[maybe_unused]] const backend_type backend, [[maybe_unused]] int &argc, [[maybe_unused]] char **argv) {
    PLSSVM_ASSERT(backend != backend_type::automatic, "The automatic backend may never be initialized!");
    // Note: must be implemented for the backends that need environmental setup
    // only have to perform special initialization steps for the HPX backend
#if defined(PLSSVM_HAS_HPX_BACKEND)
    if (backend == backend_type::hpx) {
        ::hpx::start(nullptr, argc, argv);
    }
#endif
}

/**
 * @brief Finalize the @p backend.
 * @param[in] backend  the backend to finalize
 */
inline void finalize_backend([[maybe_unused]] const backend_type backend) {
    PLSSVM_ASSERT(backend != backend_type::automatic, "The automatic backend may never be finalized!");
    // Note: must be implemented for the backends that need environmental setup
    // only have to perform special initialization steps for the HPX backend
#if defined(PLSSVM_HAS_HPX_BACKEND)
    if (backend == backend_type::hpx) {
        ::hpx::post([] { ::hpx::finalize(); });
    }
#endif
}

/**
 * @brief Try to initialize all @p backends.
 * @tparam Args the type of the arguments used for the initialization
 * @param[in] backends the backends to initialize
 * @param [in,out] args the arguments used for the initialization
 * @throws plssvm::environment_exception if any of the provided @p backends isn't currently available
 * @throws plssvm::environment_exception if the automatic backend is provided in @p backends
 * @throws plssvm::environment_exception if one of the provided @p backends has already been initialized
 * @throws plssvm::environment_exception if one of the provided @p backends has already been finalized
 */
template <typename... Args>
inline void initialize_impl(const std::vector<backend_type> &backends, Args &...args) {
    // check if the provided backends are currently available
    const std::vector<backend_type> available_backends = list_available_backends();
    for (const backend_type backend : backends) {
        // check if the backend is available
        if (!::plssvm::detail::contains(available_backends, backend)) {
            throw environment_exception{ fmt::format("The provided backend {} is currently not available and, therefore, can't be initialized! Available backends are: [{}].", backend, fmt::join(available_backends, ", ")) };
        }
    }

    // try to initialize the backend environments
    for (const backend_type backend : backends) {
        // the automatic backend cannot be initialized
        if (backend == backend_type::automatic) {
            throw environment_exception{ "The automatic backend cannot be initialized!" };
        }

        // check the status of the current backend
        switch (get_backend_status(backend)) {
            case status::uninitialized:
                // currently uninitialized -> initialize backend
                detail::initialize_backend(backend, args...);
                break;
            case status::initialized:
                // backend already initialized -> throw exception
                throw environment_exception{ fmt::format("The backend {} has already been initialized!", backend) };
            case status::finalized:
                // backend already finalized -> throw exception
                throw environment_exception{ fmt::format("The backend {} has already been finalized!", backend) };
            case status::unnecessary:
                // no initialization or finalization necessary -> do nothing
                break;
        }
    }
}

/**
 * @brief Remove all backends from @p backends for which the `get_backend_status()` function doesn't return @p s.
 *        Additionally, always removes the automatic backend.
 * @param[in,out] backends the backends to filter
 * @param[in] s the filter to use
 */
inline void get_filtered_backends(std::vector<backend_type> &backends, const status s) {
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

//****************************************************************************//
//                 initialization and finalization functions                  //
//****************************************************************************//

/**
 * @brief Initialize all of the provided @p backends.
 * @param[in] backends all backends that should be initialized
 * @throws plssvm::environment_exception if any of the provided @p backends isn't currently available
 * @throws plssvm::environment_exception if the automatic backend is provided in @p backends
 * @throws plssvm::environment_exception if one of the provided @p backends has already been initialized
 * @throws plssvm::environment_exception if one of the provided @p backends has already been finalized
 */
inline void initialize(const std::vector<backend_type> &backends) {
    detail::initialize_impl(backends);
}

/**
 * @brief Initialize all **available** backends.
 * @details Only initializes backends that are currently uninitialized.
 * @return the initialized backends (`[[nodiscard]]`)
 */
inline std::vector<backend_type> initialize() {
    // get all available backends
    std::vector<backend_type> backends = list_available_backends();
    // only initialize currently uninitialized backends; remove the automatic backend
    detail::get_filtered_backends(backends, status::uninitialized);
    // initialize the remaining backends
    detail::initialize_impl(backends);
    return backends;
}

/**
 * @brief Initialize all of the provided @p backends using the command line arguments @p argc and @p argv.
 * @param[in,out] argc the number of provided command line arguments
 * @param[in,out] argv the provided command line arguments
 * @param[in] backends all backends that should be initialized
 * @throws plssvm::environment_exception if any of the provided @p backends isn't currently available
 * @throws plssvm::environment_exception if the automatic backend is provided in @p backends
 * @throws plssvm::environment_exception if one of the provided @p backends has already been initialized
 * @throws plssvm::environment_exception if one of the provided @p backends has already been finalized
 */
inline void initialize(int &argc, char **argv, const std::vector<backend_type> &backends) {
    detail::initialize_impl(backends, argc, argv);
}

/**
 * @brief Initialize all **available** backends.
 * @details Only initializes backends that are currently uninitialized.
 * @param[in,out] argc the number of provided command line arguments
 * @param[in,out] argv the provided command line arguments
 * @return the initialized backends (`[[nodiscard]]`)
 */
inline std::vector<backend_type> initialize(int &argc, char **argv) {
    // get all available backends
    std::vector<backend_type> backends = list_available_backends();
    // only initialize currently uninitialized backends; remove the automatic backend
    detail::get_filtered_backends(backends, status::uninitialized);
    // initialize the remaining backends
    detail::initialize_impl(backends, argc, argv);
    return backends;
}

/**
 * @brief Try to finalize all @p backends.
 * @param[in] backends the backends to finalize
 * @throws plssvm::environment_exception if the automatic backend is provided in @p backends
 * @throws plssvm::environment_exception if one of the provided @p backends hasn't been initialized yet
 * @throws plssvm::environment_exception if one of the provided @p backends has already been finalized
 */
inline void finalize(const std::vector<backend_type> &backends) {
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

/**
 * @brief Finalize all **available** backends.
 * @details Only finalizes backends that are currently initialized.
 * @return the finalized backends (`[[nodiscard]]`)
 */
inline std::vector<backend_type> finalize() {
    // get all available backends
    std::vector<backend_type> backends = list_available_backends();
    // only finalize currently initialized backends; remove the automatic backend
    detail::get_filtered_backends(backends, status::initialized);
    // finalize the remaining backends
    finalize(backends);
    return backends;
}

//****************************************************************************//
//                     custom scope guard implementation                      //
//****************************************************************************//

/**
 * @brief A scope guard to initialize and automatically finalize all necessary environments in order for all PLSSVM backends to work properly.
 */
class [[nodiscard]] scope_guard {
  public:
    /**
     * @copydoc initialize()
     */
    scope_guard() {
        backends_ = initialize();
    }

    /**
     * @copydoc initialize(const std::vector<backend_type> &)
     */
    explicit scope_guard(std::vector<backend_type> backends) :
        backends_{ std::move(backends) } {
        initialize(backends_);
    }

    /**
     * @copydoc initialize(int &, char **)
     */
    scope_guard(int &argc, char **argv) {
        backends_ = initialize(argc, argv);
    }

    /**
     * @copydoc initialize(int &, char **, const std::vector<backend_type> &)
     */
    scope_guard(int &argc, char **argv, std::vector<backend_type> backends) :
        backends_{ std::move(backends) } {
        initialize(argc, argv, backends_);
    }

    /**
     * @brief Delete copy-constructor since a scope_guard is a move-only type.
     */
    scope_guard(const scope_guard &) = delete;
    /**
     * @brief Default move-constructor.
     */
    scope_guard(scope_guard &&) noexcept = default;
    /**
     * @brief Delete copy-assignment operator since a scope_guard is a move-only type.
     * @return `*this`
     */
    scope_guard &operator=(const scope_guard &) = delete;
    /**
     * @brief Default move-assignment operator.
     * @return `*this`
     */
    scope_guard &operator=(scope_guard &&) noexcept = default;

    /**
     * @brief Return the backends that are initialized and finalized using this scope_guard.
     * @return the initialized and finalized backends (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<backend_type> &backends() const noexcept {
        return backends_;
    }

    /**
     * @brief Finalize all previously initialized backends.
     */
    ~scope_guard() {
        finalize(backends_);
    }

  private:
    /// The backends that should be initialized IF it is necessary for them or all available if empty.
    std::vector<backend_type> backends_{};
};

}  // namespace plssvm::environment

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<plssvm::environment::status> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_ENVIRONMENT_HPP_
