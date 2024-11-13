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

#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    #include "Kokkos_Core.hpp"  // Kokkos::is_initialized, Kokkos::is_finalized, Kokkos::initialize, Kokkos::finalize
#endif

#include "plssvm/backend_types.hpp"   // plssvm::backend_type, plssvm::list_available_backends
#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains

#include <vector>  // std::vector

namespace plssvm::environment {

/**
 * @brief Check, whether the environments have already been initialized correctly.
 * @return `true` if the environments are initialized correctly, `false` otherwise
 */
[[nodiscard]] inline bool is_initialized() {
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    return Kokkos::is_initialized();
#else
    return true;
#endif
}

/**
 * @brief Check, whether the environments have been finalized correctly.
 * @return `true` if the environments are finalized correctly, `false` otherwise
 */
[[nodiscard]] inline bool is_finalized() {
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    return Kokkos::is_finalized();
#else
    return true;
#endif
}

/**
 * @brief Initialize all necessary environments in order for all PLSSVM backends to work properly.
 */
inline void initialize([[maybe_unused]] const std::vector<backend_type> &backends_to_init = list_available_backends()) {
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    if (detail::contains(backends_to_init, backend_type::automatic) || detail::contains(backends_to_init, backend_type::kokkos)) {
        Kokkos::initialize();
    }
#endif
}

/**
 * @brief Initialize all necessary environments in order for all PLSSVM backends to work properly.
 * @param[in] argc the number of provided command line arguments
 * @param[in] argv the provided command line arguments
 */
inline void initialize([[maybe_unused]] int &argc, [[maybe_unused]] char **argv, [[maybe_unused]] const std::vector<backend_type> &backends_to_init = list_available_backends()) {
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    if (detail::contains(backends_to_init, backend_type::automatic) || detail::contains(backends_to_init, backend_type::kokkos)) {
        Kokkos::initialize(argc, argv);
    }
#endif
}

/**
 * @brief Finalize all necessary environments in order for all PLSSVM backends to work properly.
 */
inline void finalize([[maybe_unused]] const std::vector<backend_type> &backends_to_init = list_available_backends()) {
#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    if (detail::contains(backends_to_init, backend_type::automatic) || detail::contains(backends_to_init, backend_type::kokkos)) {
        Kokkos::finalize();
    }
#endif
}

/**
 * @brief A scope guard to initialize and automatically finalize all necessary environments in order for all PLSSVM backends to work properly.
 */
class [[nodiscard]] scope_guard {
  public:
    /**
     * @brief If the environments are not already initialized, initialize all necessary environments in order for all PLSSVM backends to work properly
     */
    explicit scope_guard(std::vector<backend_type> backends_to_init = list_available_backends()) :
        backends_to_init_{ std::move(backends_to_init) } {
        if (!is_initialized()) {
            initialize(backends_to_init_);
        }
    }

    /**
     * @brief If the environments are not already initialized, initialize all necessary environments in order for all PLSSVM backends to work properly
     * @param[in] argc the number of provided command line arguments
     * @param[in] argv the provided command line arguments
     */
    scope_guard(int &argc, char **argv, std::vector<backend_type> backends_to_init = list_available_backends()) :
        backends_to_init_{ std::move(backends_to_init) } {
        if (!is_initialized()) {
            initialize(argc, argv, backends_to_init_);
        }
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
     * @brief If the environments are not already finalized, finalize all necessary environments in order for all PLSSVM backends to work properly.
     */
    ~scope_guard() {
        if (!is_finalized()) {
            finalize(backends_to_init_);
        }
    }

  private:
    std::vector<backend_type> backends_to_init_{};
};

}  // namespace plssvm::environment

#endif  // PLSSVM_ENVIRONMENT_HPP_
