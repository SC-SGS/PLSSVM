/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Defines all currently supported SYCL implementations.
*/

#pragma once

#include <iosfwd>  // forward declare std::ostream and std::istream

namespace plssvm::sycl_generic {

/**
* @brief Enum class for all possible SYCL kernel invocation types.
*/
enum class implementation_type {
    /** Use the available SYCL implementation. If more than one implementation is available, use PLSSVM_SYCL_BACKEND_PREFERRED_IMPLEMENTATION. */
    automatic,
    /** Use [DPC++](https://github.com/intel/llvm) as SYCL implementation. */
    dpcpp,
    /** Use [hipSYCL](https://github.com/illuhad/hipSYCL) as SYCL implementation. */
    hipsycl
};

/**
* @brief Output the @p impl type to the given output-stream @p out.
* @param[in,out] out the output-stream to write the backend type to
* @param[in] impl the SYCL implementation type
* @return the output-stream
*/
std::ostream &operator<<(std::ostream &out, implementation_type impl);

/**
* @brief Use the input-stream @p in to initialize the @p impl type.
* @param[in,out] in input-stream to extract the backend type from
* @param[in] impl the SYCL implementation type
* @return the input-stream
*/
std::istream &operator>>(std::istream &in, implementation_type &impl);

}  // namespace plssvm::sycl_generic
