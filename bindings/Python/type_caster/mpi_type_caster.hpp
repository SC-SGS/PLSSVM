/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a custom type caster for a plssvm::mpi::communicator used with mpi4py.
 */

#ifndef PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_MPI_TYPE_CASTER_HPP_
#define PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_MPI_TYPE_CASTER_HPP_
#pragma once

#include "plssvm/mpi/communicator.hpp"  // plssvm::mpi::communicator

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi/mpi.h"  // MPI_Comm, MPI_Comm_c2f, MPI_Comm_f2c, MPI_Fint
#endif

#include "pybind11/cast.h"      // pybind11::detail::type_caster
#include "pybind11/pybind11.h"  // py::module, py::isinstance, py::none, py::return_value_policy, py::error_already_set
#include "pybind11/pytypes.h"   // py::handle

namespace py = pybind11;

namespace pybind11::detail {

/**
 * @brief A custom Pybind11 type caster to convert Python object from and to a plssvm::mpi::communicator.
 */
template <>
struct type_caster<plssvm::mpi::communicator> {
  public:
    /// Specify the Python type name to which a plssvm::mpi::communicator should be converted.
    PYBIND11_TYPE_CASTER(plssvm::mpi::communicator, _("MPI_Comm"));

    /**
     * @brief Convert a plssvm::mpi::communicator to a mpi4py communicator.
     * @param[in] comm the PLSSVM MPI communicator wrapper
     * @return a Pybind11 handle to the mpi4py communicator
     */
    static py::handle cast([[maybe_unused]] const plssvm::mpi::communicator &comm, py::return_value_policy, py::handle) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
        // we have MPI enabled
        try {
            // if we can find mpi4py, we can convert a MPI_Comm to its mpi4py representation
            const py::module_ mpi4py = py::module_::import("mpi4py.MPI");
            return mpi4py.attr("Comm").attr("f2py")(MPI_Comm_c2f(static_cast<MPI_Comm>(comm))).release();
        } catch (const py::error_already_set &) {
            // we couldn't find mpi4py, so simply return None since anything else doesn't make sense
            return py::none{}.release();
        }
#else
        // we haven't MPI enabled -> return None since anything else doesn't make sense
        return py::none{}.release();
#endif
    }

    /**
     * @brief Try converting a Python object @p obj to a plssvm::mpi::communicator.
     * @param[in] obj the object to convert
     * @return `true` if the conversion was successful, `false` otherwise
     * @throws py::value_error if PLSSVM was built without MPI support, but a communicator was explicitly provided in Python
     */
    bool load([[maybe_unused]] py::handle obj, bool) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
        try {
            // check if we can find mpi4py
            const py::module_ mpi4py = py::module_::import("mpi4py.MPI");
            // we found mpi4py -> check whether a communicator was provided
            if (py::isinstance(obj, mpi4py.attr("Comm"))) {
                // we got a mpi4py communicator -> cast it to an MPI_Comm handle and create a new plssvm::mpi::communicator
                const MPI_Fint f_handle = obj.attr("py2f")().cast<MPI_Fint>();
                value = plssvm::mpi::communicator{ MPI_Comm_f2c(f_handle) };
                return true;
            } else {
                // something else was provided -> abort type casting
                return false;
            }
        } catch (const py::error_already_set &) {
            // we couldn't find mpi4py
            if (obj.is_none()) {
                // but "comm" wasn't set -> we can use our default plssvm::mpi::communicator
                value = plssvm::mpi::communicator{};
                return true;
            } else {
                // something was provided -> abort type casting
                return false;
            }
        }
#else
        // we haven't MPI enabled -> check whether the "comm" argument has been provided
        if (!obj.is_none()) {
            // "comm" has been provided -> we can't use it -> throw an exception
            throw py::value_error{ "ERROR: an MPI communicator was explicitly provided, but PLSSVM was built without support for MPI!" };
        } else {
            // "comm" was not provided -> use a default constructed plssvm::mpi::communicator that essentially does nothing
            value = plssvm::mpi::communicator{};
            return true;
        }
#endif
    }
};

}  // namespace pybind11::detail

#endif  // PLSSVM_BINDINGS_PYTHON_TYPE_CASTER_MPI_TYPE_CASTER_HPP_
