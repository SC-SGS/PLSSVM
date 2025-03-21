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
#pragma once

#include "plssvm/backend_types.hpp"          // plssvm::backend_type, plssvm::list_available_backends
#include "plssvm/detail/utility.hpp"         // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::environment_exception
#include "plssvm/mpi/environment.hpp"        // plssvm::mpi::{is_initialized, init}

#include "fmt/base.h"     // fmt::formatter
#include "fmt/format.h"   // fmt::format
#include "fmt/ostream.h"  // fmt::ostream_formatter
#include "fmt/ranges.h"   // fmt::join

#include <iosfwd>   // forward declare std::ostream and std::istream
#include <utility>  // std::move
#include <vector>   // std::vector

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
std::ostream &operator<<(std::ostream &out, status s);

/**
 * @brief Use the input-stream @p in to initialize the environment status @p s.
 * @param[in,out] in input-stream to extract the environment status from
 * @param[in] s the environment status
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, status &s);

namespace detail {

/**
 * @brief Determine the environment status based on the @p is_initialized and @p is_finalized flags.
 * @param is_initialized `true` if the respective environment has been initialized, `false` otherwise
 * @param is_finalized `true` if the respective environment has been finalized, `false` otherwise
 * @return the respective environment status (`[[nodiscard]]`)
 */
[[nodiscard]] status determine_status_from_initialized_finalized_flags(bool is_initialized, bool is_finalized);

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
[[nodiscard]] status get_backend_status(const backend_type backend);

/**
 * @brief Check whether the provided backend needs a special initialization.
 * @param[in] backend the backend to check
 * @return `true` if the backend needs a special environment initialization, `false` otherwise
 */
constexpr bool is_initialization_necessary([[maybe_unused]] const backend_type backend) {
    // Note: must be implemented for the backends that need environmental setup
    // currently false for all available backends
    return backend == backend_type::hpx || backend == backend_type::kokkos;
}

//****************************************************************************//
//         backend specific initialization and finalization functions         //
//****************************************************************************//

namespace detail {

/**
 * @brief Initialize the @p backend.
 * @param[in] backend the backend to initialize
 */
void initialize_backend(backend_type backend);

/**
 * @brief Initialize the @p backend with the provided command line arguments.
 * @param[in] backend the backend to initialize
 * @param[in,out] argc the number of command line arguments
 * @param[in,out] argv the command line arguments
 */
void initialize_backend(backend_type backend, int &argc, char **argv);

/**
 * @brief Finalize the @p backend.
 * @param[in] backend  the backend to finalize
 */
void finalize_backend(backend_type backend);

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
    // if necessary, initialize MPI
    if (!mpi::is_initialized()) {
        mpi::init(args...);
    }

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
void get_filtered_backends(std::vector<backend_type> &backends, status s);

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
void initialize(const std::vector<backend_type> &backends);

/**
 * @brief Initialize all **available** backends.
 * @details Only initializes backends that are currently uninitialized.
 * @return the initialized backends (`[[nodiscard]]`)
 */
std::vector<backend_type> initialize();

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
void initialize(int &argc, char **argv, const std::vector<backend_type> &backends);

/**
 * @brief Initialize all **available** backends.
 * @details Only initializes backends that are currently uninitialized.
 * @param[in,out] argc the number of provided command line arguments
 * @param[in,out] argv the provided command line arguments
 * @return the initialized backends (`[[nodiscard]]`)
 */
std::vector<backend_type> initialize(int &argc, char **argv);

/**
 * @brief Try to finalize all @p backends.
 * @param[in] backends the backends to finalize
 * @throws plssvm::environment_exception if the automatic backend is provided in @p backends
 * @throws plssvm::environment_exception if one of the provided @p backends hasn't been initialized yet
 * @throws plssvm::environment_exception if one of the provided @p backends has already been finalized
 */
void finalize(const std::vector<backend_type> &backends);

/**
 * @brief Finalize all **available** backends.
 * @details Only finalizes backends that are currently initialized.
 * @return the finalized backends (`[[nodiscard]]`)
 */
std::vector<backend_type> finalize();

//****************************************************************************//
//                     custom scope guard implementation                      //
//****************************************************************************//

/**
 * @brief A scope guard to initialize and automatically finalize all necessary environments in order for all PLSSVM backends to work properly.
 */
class [[nodiscard]] scope_guard {
  public:
    /**
     * @brief Initialize all **available** backends.
     * @details Only initializes backends that are currently uninitialized.
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
     * @brief Initialize all **available** backends.
     * @details Only initializes backends that are currently uninitialized.
     * @param[in,out] argc the number of provided command line arguments
     * @param[in,out] argv the provided command line arguments
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
