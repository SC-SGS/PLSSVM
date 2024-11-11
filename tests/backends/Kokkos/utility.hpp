/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Determine the execution spaces available for tests with the Kokkos backend.
*/

#ifndef PLSSVM_TESTS_BACKENDS_KOKKOS_UTILITY_HPP_
#define PLSSVM_TESTS_BACKENDS_KOKKOS_UTILITY_HPP_
#pragma once

namespace util {

/**
 * @brief Determine which execution spaces can be tested based on the available Kokkos::ExecutionSpaces and PLSSVM target platforms.
 * @return the available execution spaces for testing (`[[nodiscard]]`)
 */
[[nodiscard]] constexpr auto constexpr_available_execution_spaces_to_test() {
    return std::array{
#if defined(KOKKOS_ENABLE_CUDA) && defined(PLSSVM_HAS_NVIDIA_TARGET)  // for Kokkos::Cuda, an NVIDIA target must be available
        plssvm::kokkos::execution_space::cuda,
#endif
#if defined(KOKKOS_ENABLE_HIP) && (defined(PLSSVM_HAS_NVIDIA_TARGET) || defined(PLSSVM_HAS_AMD_TARGET))  // for Kokkos::HIP, an NVIDIA or AMD target must be available
        plssvm::kokkos::execution_space::hip,
#endif
#if defined(KOKKOS_ENABLE_SYCL)  // for Kokkos::SYCL, any target is ok
        plssvm::kokkos::execution_space::sycl,
#endif
#if defined(KOKKOS_ENABLE_HPX) && defined(PLSSVM_HAS_CPU_TARGET)  // for Kokkos::Experimental::HPX, a CPU target must be available
        plssvm::kokkos::execution_space::hpx,
#endif
#if defined(KOKKOS_ENABLE_OPENMP) && defined(PLSSVM_HAS_CPU_TARGET)  // for Kokkos::OpenMP, a CPU target must be available
        plssvm::kokkos::execution_space::openmp,
#endif
#if defined(KOKKOS_ENABLE_OPENMPTARGET)  // for Kokkos::Experimental::OpenMPTarget,any target is ok // TODO: implement correctly based on allowed target platforms
        plssvm::kokkos::execution_space::openmp_target,
#endif
#if defined(KOKKOS_ENABLE_OPENACC)  // for Kokkos::Experimental::OpenACC,any target is ok // TODO: implement correctly based on allowed target platforms
        plssvm::kokkos::execution_space::openacc,
#endif
#if defined(KOKKOS_ENABLE_THREADS) && defined(PLSSVM_HAS_CPU_TARGET)  // for Kokkos::Threads, a CPU target must be available
        plssvm::kokkos::execution_space::threads,
#endif
#if defined(KOKKOS_ENABLE_SERIAL) && defined(PLSSVM_HAS_CPU_TARGET)  // for Kokkos::Serial, a CPU target must be available
        plssvm::kokkos::execution_space::serial,
#endif
    };
}

/**
 * @brief Uninstantiated base type to create a `std::tuple` containing all available `kokkos_csvm_test_type` types.
 */
template <template <plssvm::kokkos::execution_space> typename, typename>
struct create_kokkos_test_tuple_impl;

/**
 * @brief Helper struct to create a `std::tuple` containing all available `test_type` types by iterating over the `std::array` of
 *        `plssvm::kokkos::execution_space` values as returned by `plssvm::kokkos::detail::constexpr_available_execution_spaces()`.
 * @tparam test_type the test type to instantiate
 * @tparam Is the indices to index the `std::array`
 */
template <template <plssvm::kokkos::execution_space> typename test_type, std::size_t... Is>
struct create_kokkos_test_tuple_impl<test_type, std::index_sequence<Is...>> {
    /// The array containing all available execution spaces.
    constexpr static auto array = constexpr_available_execution_spaces_to_test();
    /// The resulting variant type.
    using type = std::tuple<test_type<array[Is]>...>;
};

/**
 * @brief Create a `std::tuple` containing all available `test_type` types by iterating over the `std::array` of
 *        `plssvm::kokkos::execution_space` values as returned by `plssvm::kokkos::detail::constexpr_available_execution_spaces()`.
 * @tparam test_type the test type to instantiate
 */
template <template <plssvm::kokkos::execution_space> typename test_type>
struct create_kokkos_test_tuple {
    /// The number of types in the final variant.
    constexpr static std::size_t N = constexpr_available_execution_spaces_to_test().size();
    /// The final tuple type.
    using type = typename create_kokkos_test_tuple_impl<test_type, std::make_index_sequence<N>>::type;
};

/**
 * @brief Shorthand for the `typename create_kokkos_test_tuple<...>::type` type.
 */
template <template <plssvm::kokkos::execution_space> typename test_type>
using create_kokkos_test_tuple_t = typename create_kokkos_test_tuple<test_type>::type;


}

#endif  // PLSSVM_TESTS_BACKENDS_KOKKOS_UTILITY_HPP_
