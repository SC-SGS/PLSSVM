/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the OpenMP backend.
 */

#ifndef PLSSVM_BACKENDS_OPENMP_CSVM_HPP_
#define PLSSVM_BACKENDS_OPENMP_CSVM_HPP_
#pragma once

#include "plssvm/constants.hpp"             // plssvm::real_type
#include "plssvm/detail/memory_size.hpp"    // plssvm::detail::memory_size
#include "plssvm/detail/move_only_any.hpp"  // plssvm::detail::move_only_any
#include "plssvm/detail/type_traits.hpp"    // PLSSVM_REQUIRES, plssvm::detail::is_one_type_of
#include "plssvm/matrix.hpp"                // plssvm::aos_matrix
#include "plssvm/mpi/communicator.hpp"      // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"             // plssvm::parameter, plssvm::detail::has_only_parameter_named_args_v
#include "plssvm/solver_types.hpp"          // plssvm::solver_type
#include "plssvm/svm/csvc.hpp"              // plssvm::csvc
#include "plssvm/svm/csvm.hpp"              // plssvm::csvm, plssvm::detail::csvm_backend_exists
#include "plssvm/svm/csvr.hpp"              // plssvm::csvr
#include "plssvm/target_platforms.hpp"      // plssvm::target_platform

#include <cstddef>      // std::size_t
#include <memory>       // std::addressof
#include <optional>     // std::optional
#include <type_traits>  // std::true_type
#include <utility>      // std::forward, std::pair
#include <vector>       // std::vector

namespace plssvm {

namespace openmp {

/**
 * @brief A C-SVM implementation using OpenMP as backend.
 */
class csvm : virtual public ::plssvm::csvm {
  public:
    /**
     * @brief Construct a new C-SVM using the OpenMP backend on the @p target platform.
     * @param[in] target the target platform used for this C-SVM
     * @throws plssvm::exception all exceptions thrown in the base class constructor
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    explicit csvm(target_platform target = target_platform::automatic);

    /**
     * @copydoc plssvm::csvm::csvm(const plssvm::csvm &)
     */
    csvm(const csvm &) = delete;
    /**
     * @copydoc plssvm::csvm::csvm(plssvm::csvm &&) noexcept
     */
    csvm(csvm &&) noexcept = default;
    /**
     * @copydoc plssvm::csvm::operator=(const plssvm::csvm &)
     */
    csvm &operator=(const csvm &) = delete;

    /**
     * @brief Correctly implement the move-assignment operator in presence of a virtual base class.
     * @details Calls the base class move-assignment operator. Afterwards, moves the additional `openmp::csvm` members.
     * @param[in,out] other the other C-SVM to move from
     * @return `*this`
     */
    csvm &operator=(csvm &&other) noexcept {
        if (this != std::addressof(other)) {
            ::plssvm::csvm::operator=(std::move(other));
        }
        return *this;
    }

    /**
     * @brief Default destructor since the copy and move constructors and copy- and move-assignment operators are defined.
     */
    ~csvm() override = 0;

    /**
     * @copydoc plssvm::csvm::num_available_devices
     * @note On the CPU, only one device will ever be available.
     */
    [[nodiscard]] std::size_t num_available_devices() const noexcept override {
        return 1;
    }

  protected:
    /**
     * @copydoc plssvm::csvm::get_device_memory
     */
    [[nodiscard]] std::vector<::plssvm::detail::memory_size> get_device_memory() const final;
    /**
     * @copydoc plssvm::csvm::get_max_mem_alloc_size
     */
    [[nodiscard]] std::vector<::plssvm::detail::memory_size> get_max_mem_alloc_size() const final;
    /**
     * @copydoc plssvm::csvm::get_local_memory
     */
    [[nodiscard]] std::vector<std::optional<::plssvm::detail::memory_size>> get_local_memory() const final;

    //***************************************************//
    //                        fit                        //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::assemble_kernel_matrix
     */
    [[nodiscard]] std::vector<::plssvm::detail::move_only_any> assemble_kernel_matrix(solver_type solver, const parameter &params, const soa_matrix<real_type> &A, const std::vector<real_type> &q_red, real_type QA_cost) const final;
    /**
     * @copydoc plssvm::csvm::blas_level_3
     */
    void blas_level_3(solver_type solver, real_type alpha, const std::vector<::plssvm::detail::move_only_any> &A, const soa_matrix<real_type> &B, real_type beta, soa_matrix<real_type> &C) const final;

    //***************************************************//
    //                   predict, score                  //
    //***************************************************//
    /**
     * @copydoc plssvm::csvm::predict_values
     */
    [[nodiscard]] aos_matrix<real_type> predict_values(const parameter &params, const soa_matrix<real_type> &support_vectors, const aos_matrix<real_type> &alpha, const std::vector<real_type> &rho, soa_matrix<real_type> &w, const soa_matrix<real_type> &predict_points) const final;
};

/**
 * @brief Create a C-SVC using the OpenMP backend.
 * @details Inherits all functionality either from the `plssvm::csvc` or `plssvm::openmp::csvm` classes.
 */
class csvc : public ::plssvm::csvc,
             public ::plssvm::openmp::csvm {
  public:
    /**
     * @brief Construct a new C-SVC using the OpenMP backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    explicit csvc(const parameter params) :
        ::plssvm::csvm{ mpi::communicator{}, params } { }

    /**
     * @brief Construct a new C-SVC using the OpenMP backend with the parameters given through @p params.
     * @param[in] comm the used MPI communicator
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    explicit csvc(mpi::communicator comm, const parameter params) :
        ::plssvm::csvm{ std::move(comm), params } { }

    /**
     * @brief Construct a new C-SVC using the OpenMP backend on the @p target platform with the parameters given through @p params.
     * @param[in] target the target platform used for this C-SVC
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    csvc(const target_platform target, const parameter params) :
        ::plssvm::csvm{ mpi::communicator{}, params },
        ::plssvm::openmp::csvm{ target } { }

    /**
     * @brief Construct a new C-SVC using the OpenMP backend on the @p target platform with the parameters given through @p params.
     * @param[in] comm the used MPI communicator
     * @param[in] target the target platform used for this C-SVC
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    csvc(mpi::communicator comm, const target_platform target, const parameter params) :
        ::plssvm::csvm{ std::move(comm), params },
        ::plssvm::openmp::csvm{ target } { }

    /**
     * @brief Construct a new C-SVC using the OpenMP backend and the optionally provided @p named_args.
     * @param[in] named_args the additional optional named arguments
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvc(Args &&...named_args) :
        ::plssvm::csvm{ mpi::communicator{}, std::forward<Args>(named_args)... },
        ::plssvm::openmp::csvm{} { }

    /**
     * @brief Construct a new C-SVC using the OpenMP backend and the optionally provided @p named_args.
     * @param[in] comm the used MPI communicator
     * @param[in] named_args the additional optional named arguments
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvc(mpi::communicator comm, Args &&...named_args) :
        ::plssvm::csvm{ std::move(comm), std::forward<Args>(named_args)... },
        ::plssvm::openmp::csvm{} { }

    /**
     * @brief Construct a new C-SVC using the OpenMP backend on the @p target platform and the optionally provided @p named_args.
     * @param[in] target the target platform used for this C-SVC
     * @param[in] named_args the additional optional named-parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvc(const target_platform target, Args &&...named_args) :
        ::plssvm::csvm{ mpi::communicator{}, std::forward<Args>(named_args)... },
        ::plssvm::openmp::csvm{ target } { }

    /**
     * @brief Construct a new C-SVC using the OpenMP backend on the @p target platform and the optionally provided @p named_args.
     * @param[in] comm the used MPI communicator
     * @param[in] target the target platform used for this C-SVC
     * @param[in] named_args the additional optional named-parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    csvc(mpi::communicator comm, const target_platform target, Args &&...named_args) :
        ::plssvm::csvm{ std::move(comm), std::forward<Args>(named_args)... },
        ::plssvm::openmp::csvm{ target } { }
};

/**
 * @brief Create a C-SVR using the OpenMP backend.
 * @details Inherits all functionality either from the `plssvm::csvr` or `plssvm::openmp::csvm` classes.
 */
class csvr : public ::plssvm::csvr,
             public ::plssvm::openmp::csvm {
  public:
    /**
     * @brief Construct a new C-SVR using the OpenMP backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    explicit csvr(const parameter params) :
        ::plssvm::csvm{ mpi::communicator{}, params } { }

    /**
     * @brief Construct a new C-SVR using the OpenMP backend with the parameters given through @p params.
     * @param[in] comm the used MPI communicator
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    csvr(mpi::communicator comm, const parameter params) :
        ::plssvm::csvm{ std::move(comm), params } { }

    /**
     * @brief Construct a new C-SVR using the OpenMP backend on the @p target platform with the parameters given through @p params.
     * @param[in] target the target platform used for this C-SVR
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    csvr(const target_platform target, const parameter params) :
        ::plssvm::csvm{ mpi::communicator{}, params },
        ::plssvm::openmp::csvm{ target } { }

    /**
     * @brief Construct a new C-SVR using the OpenMP backend on the @p target platform with the parameters given through @p params.
     * @param[in] comm the used MPI communicator
     * @param[in] target the target platform used for this C-SVR
     * @param[in] params struct encapsulating all possible SVM parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    csvr(mpi::communicator comm, const target_platform target, const parameter params) :
        ::plssvm::csvm{ std::move(comm), params },
        ::plssvm::openmp::csvm{ target } { }

    /**
     * @brief Construct a new C-SVR using the OpenMP backend and the optionally provided @p named_args.
     * @param[in] named_args the additional optional named arguments
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvr(Args &&...named_args) :
        ::plssvm::csvm{ mpi::communicator{}, std::forward<Args>(named_args)... },
        ::plssvm::openmp::csvm{} { }

    /**
     * @brief Construct a new C-SVR using the OpenMP backend and the optionally provided @p named_args.
     * @param[in] comm the used MPI communicator
     * @param[in] named_args the additional optional named arguments
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvr(mpi::communicator comm, Args &&...named_args) :
        ::plssvm::csvm{ std::move(comm), std::forward<Args>(named_args)... },
        ::plssvm::openmp::csvm{} { }

    /**
     * @brief Construct a new C-SVR using the OpenMP backend on the @p target platform and the optionally provided @p named_args.
     * @param[in] target the target platform used for this C-SVR
     * @param[in] named_args the additional optional named-parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    explicit csvr(const target_platform target, Args &&...named_args) :
        ::plssvm::csvm{ mpi::communicator{}, std::forward<Args>(named_args)... },
        ::plssvm::openmp::csvm{ target } { }

    /**
     * @brief Construct a new C-SVR using the OpenMP backend on the @p target platform and the optionally provided @p named_args.
     * @param[in] comm the used MPI communicator
     * @param[in] target the target platform used for this C-SVR
     * @param[in] named_args the additional optional named-parameters
     * @throws plssvm::exception all exceptions thrown in the base class constructors
     */
    template <typename... Args, PLSSVM_REQUIRES(::plssvm::detail::has_only_parameter_named_args_v<Args...>)>
    csvr(mpi::communicator comm, const target_platform target, Args &&...named_args) :
        ::plssvm::csvm{ std::move(comm), std::forward<Args>(named_args)... },
        ::plssvm::openmp::csvm{ target } { }
};

}  // namespace openmp

namespace detail {

/**
 * @brief Sets the `value` to `true` since C-SVMs (C-SVCs, C-SVRs) using the OpenMP backend are available.
 */
template <typename T>
struct csvm_backend_exists<T, std::enable_if_t<is_one_type_of_v<T, openmp::csvm, openmp::csvc, openmp::csvr>>> : std::true_type { };

}  // namespace detail

}  // namespace plssvm

#endif  // PLSSVM_BACKENDS_OPENMP_CSVM_HPP_
