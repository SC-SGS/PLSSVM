/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines an enumeration holding all possible backends.
 *        Can also include backends not available on the current target platform.
 */

#ifndef PLSSVM_BACKEND_TYPES_HPP_
#define PLSSVM_BACKEND_TYPES_HPP_
#pragma once

#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl::implementation_type
#include "plssvm/detail/type_traits.hpp"                 // plssvm::detail::remove_cvref_t
#include "plssvm/target_platforms.hpp"                   // plssvm::list_available_target_platforms

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <iosfwd>  // forward declare std::ostream and std::istream
#include <vector>  // std::vector

namespace plssvm {

/**
 * @brief Enum class for all possible backend types.
 * @note All different SYCL implementations have the same backend type "sycl".
 */
enum class backend_type {
    /** The default backend. Depends on the specified target platform. */
    automatic,
    /** [OpenMP](https://www.openmp.org/) to target CPUs only (currently no OpenMP target offloading support). */
    openmp,
    /** [CUDA](https://developer.nvidia.com/cuda-zone) to target NVIDIA GPUs only. */
    cuda,
    /** [HIP](https://github.com/ROCm-Developer-Tools/HIP) to target AMD and NVIDIA GPUs. */
    hip,
    /** [OpenCL](https://www.khronos.org/opencl/) to target CPUs and GPUs from different vendors. */
    opencl,
    /** [SYCL](https://www.khronos.org/sycl/) to target CPUs and GPUs from different vendors. Currently tested SYCL implementations are [DPC++](https://github.com/intel/llvm) and [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp) (formerly known as hipSYCL). */
    sycl
};

/**
 * @brief Return a list of all currently available backends.
 * @details Only backends that where found during the CMake configuration are available.
 * @return a list of the available backends (`[[nodiscard]]`)
 */
[[nodiscard]] std::vector<backend_type> list_available_backends();

/**
 * @brief Returns the default backend (if plssvm::backend_type::automatic is used) given the backend and target platform lists.
 * @param[in] available_backends list of backends; if no backends are provided, queries all available backends
 * @param[in] available_target_platforms list of target platforms; if no target platforms are provided, queries all available target platforms
 * @return the default backend given the backend and target platform lists (`[[nodiscard]]`)
 */
[[nodiscard]] backend_type determine_default_backend(const std::vector<backend_type> &available_backends = list_available_backends(),
                                                     const std::vector<target_platform> &available_target_platforms = list_available_target_platforms());

/**
 * @brief Output the @p backend to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the backend type to
 * @param[in] backend the backend type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, backend_type backend);

/**
 * @brief Use the input-stream @p in to initialize the @p backend type.
 * @param[in,out] in input-stream to extract the backend type from
 * @param[in] backend the backend type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, backend_type &backend);

/// @cond Doxygen_suppress
// clang-format off
// Forward declare all possible C-SVMs.
namespace openmp { class csvm; }
namespace cuda { class csvm; }
namespace hip { class csvm; }
namespace opencl { class csvm; }
namespace adaptivecpp { class csvm; }
namespace dpcpp { class csvm; }
// clang-format on

namespace detail {

/**
 * @brief No `value` member variable if anything other than a C-SVM has been provided.
 */
template <typename T>
struct csvm_to_backend_type {};

/**
 * @brief Sets the `value` to `plssvm::backend_type::openmp` for the OpenMP C-SVM.
 */
template <>
struct csvm_to_backend_type<openmp::csvm> {
    /// The enum value representing the OpenMP backend.
    static constexpr backend_type value = backend_type::openmp;
};
/**
 * @brief Sets the `value` to `plssvm::backend_type::cuda` for the CUDA C-SVM.
 */
template <>
struct csvm_to_backend_type<cuda::csvm> {
    /// The enum value representing the CUDA backend.
    static constexpr backend_type value = backend_type::cuda;
};
/**
 * @brief Sets the `value` to `plssvm::backend_type::hip` for the HIP C-SVM.
 */
template <>
struct csvm_to_backend_type<hip::csvm> {
    /// The enum value representing the HIP backend.
    static constexpr backend_type value = backend_type::hip;
};
/**
 * @brief Sets the `value` to `plssvm::backend_type::opencl` for the OpenCL C-SVM.
 */
template <>
struct csvm_to_backend_type<opencl::csvm> {
    /// The enum value representing the OpenCL backend.
    static constexpr backend_type value = backend_type::opencl;
};
/**
 * @brief Sets the `value` to `plssvm::backend_type::sycl` for the SYCL C-SVM using AdaptiveCpp as SYCL implementation.
 * @details Also sets a member variable `impl` to the value `plssvm::sycl::implementation_type::adaptivecpp` (only present for SYCL backends!).
 */
template <>
struct csvm_to_backend_type<adaptivecpp::csvm> {
    /// The enum value representing the SYCL (AdaptiveCpp) backend.
    static constexpr backend_type value = backend_type::sycl;
    /// The enum value representing the SYCL implementation for the (AdaptiveCpp) SYCL backend.
    static constexpr sycl::implementation_type impl = sycl::implementation_type::adaptivecpp;
};
/**
 * @brief Sets the `value` to `plssvm::backend_type::sycl` for the SYCL C-SVM using DPC++ as SYCL implementation.
 * @details Also sets a member variable `impl` to the value `plssvm::sycl::implementation_type::dpcpp` (only present for SYCL backends!).
 */
template <>
struct csvm_to_backend_type<dpcpp::csvm> {
    /// The enum value representing the SYCL (DPC++) backend.
    static constexpr backend_type value = backend_type::sycl;
    /// The enum value representing the SYCL implementation for the (DPC++) SYCL backend.
    static constexpr sycl::implementation_type impl = sycl::implementation_type::dpcpp;
};

}  // namespace detail
/// @endcond

/**
 * @brief Get the plssvm::backend_type of the C-SVM class of type @p T. Ignores all top-level const, volatile, and reference qualifiers.
 * @details Provides a member variable `value` if @p T is a valid C-SVM.
 *          If @p T is a SYCL C-SVM, an additional member variable `impl` is provided showing the used SYCL implementation.
 * @tparam T the type of the C-SVM to get the backend type from
 */
template <typename T>
struct csvm_to_backend_type : detail::csvm_to_backend_type<detail::remove_cvref_t<T>> {};
/**
 * @copydoc plssvm::csvm_to_backend_type
 * @details A shorthand for `plssvm::csvm_to_backend_type::value`.
 */
template <typename T>
constexpr backend_type csvm_to_backend_type_v = csvm_to_backend_type<T>::value;

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::backend_type> : fmt::ostream_formatter {};

#endif  // PLSSVM_BACKEND_TYPES_HPP_