/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions used for creating the Pybind11 Python bindings for the C-SVM classes.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_SVM_UTILITY_HPP_
#define PLSSVM_BINDINGS_PYTHON_SVM_UTILITY_HPP_
#pragma once

#include "plssvm/backend_types.hpp"                        // plssvm::backend_type
#include "plssvm/backends/Kokkos/execution_spaces.hpp"     // plssvm::kokkos::execution_space
#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"  // plssvm::sycl::data_parallel_kernel
#include "plssvm/backends/SYCL/implementation_types.hpp"   // plssvm::sycl::implementation_type
#include "plssvm/csvm_factory.hpp"                         // plssvm::make_csvm
#include "plssvm/mpi/communicator.hpp"                     // plssvm::mpi::communicator
#include "plssvm/parameter.hpp"                            // plssvm::parameter, named arguments
#include "plssvm/target_platforms.hpp"                     // plssvm::target_platform

#include "bindings/Python/utility.hpp"  // plssvm::bindings::python::util::check_kwargs_for_correctness

#include "pybind11/pybind11.h"  // py::kwargs

#include <memory>   // std::unique_ptr
#include <utility>  // std::move

namespace py = pybind11;

namespace plssvm::bindings::python::util {

/**
 * @brief Assemble a C-SVM (C-SVC or C-SVR based on the template parameter @p csvm_type) using the provided parameters.
 * @tparam csvm_type the type of the C-SVM to create
 * @param[in] backend the C-SVM backend to instantiate
 * @param[in] target the target platform to run on
 * @param[in] params the SVM hyper-parameter used to train the C-SVM
 * @param[in] comm the MPI communicator
 * @param[in] optional_args optional arguments used by some C-SVMs
 * @return the created C-SVM (`[[nodiscard]]`)
 */
template <typename csvm_type>
[[nodiscard]] inline std::unique_ptr<csvm_type> assemble_csvm(const plssvm::backend_type backend, const plssvm::target_platform target, const plssvm::parameter &params, plssvm::mpi::communicator comm, const py::kwargs &optional_args) {
    // check keyword arguments
    plssvm::bindings::python::util::check_kwargs_for_correctness(optional_args, { "foo", "sycl_implementation_type", "sycl_data_parallel_kernel", "kokkos_execution_space" });

    if (backend == plssvm::backend_type::sycl) {
        // parse SYCL specific keyword arguments
        plssvm::sycl::implementation_type impl_type = plssvm::sycl::implementation_type::automatic;
        if (optional_args.contains("sycl_implementation_type")) {
            impl_type = optional_args["sycl_implementation_type"].cast<plssvm::sycl::implementation_type>();
        }
        plssvm::sycl::data_parallel_kernel data_parallel_kernel_type = plssvm::sycl::data_parallel_kernel::automatic;
        if (optional_args.contains("sycl_data_parallel_kernel")) {
            data_parallel_kernel_type = optional_args["sycl_data_parallel_kernel"].cast<plssvm::sycl::data_parallel_kernel>();
        }

        return plssvm::make_csvm<csvm_type>(backend, std::move(comm), target, params, plssvm::sycl_implementation_type = impl_type, plssvm::sycl_data_parallel_kernel = data_parallel_kernel_type);
    } else if (backend == plssvm::backend_type::kokkos) {
        // parse Kokkos specific keyword arguments
        plssvm::kokkos::execution_space space = plssvm::kokkos::execution_space::automatic;
        if (optional_args.contains("kokkos_execution_space")) {
            space = optional_args["kokkos_execution_space"].cast<plssvm::kokkos::execution_space>();
        }

        return plssvm::make_csvm<csvm_type>(backend, std::move(comm), target, params, plssvm::kokkos_execution_space = space);
    } else {
        return plssvm::make_csvm<csvm_type>(backend, std::move(comm), target, params);
    }
}

}  // namespace plssvm::bindings::python::util

#endif  // PLSSVM_BINDINGS_PYTHON_SVM_UTILITY_HPP_
