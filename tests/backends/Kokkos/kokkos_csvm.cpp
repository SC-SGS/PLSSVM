/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the functionality related to the Kokkos backend.
 */

#include "plssvm/backends/Kokkos/csvm.hpp"             // plssvm::kokkos::csvm
#include "plssvm/backends/Kokkos/detail/utility.hpp"   // plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping
#include "plssvm/backends/Kokkos/exceptions.hpp"       // plssvm::kokkos::backend_exception
#include "plssvm/backends/Kokkos/execution_space.hpp"  // plssvm::kokkos::execution_space
#include "plssvm/detail/type_list.hpp"                 // plssvm::detail::label_type_list
#include "plssvm/kernel_function_types.hpp"            // plssvm::kernel_function_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter
#include "plssvm/target_platforms.hpp"                 // plssvm::target_platform, plssvm::list_available_target_platforms

#include "tests/backends/generic_csvm_tests.hpp"      // generic CSVM tests to instantiate
#include "tests/backends/generic_gpu_csvm_tests.hpp"  // generic GPU CSVM tests to instantiate
#include "tests/backends/Kokkos/mock_kokkos_csvm.hpp"
#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "tests/naming.hpp"              // naming::test_parameter_to_name
#include "tests/types_to_test.hpp"       // util::{cartesian_type_product_t, combine_test_parameters_gtest_t}
#include "tests/utility.hpp"             // util::redirect_output

#include "gtest/gtest.h"  // TEST_F, EXPECT_NO_THROW, INSTANTIATE_TYPED_TEST_SUITE_P, ::testing::Test

#include <array>    // std::array
#include <cstddef>  // std::size_t
#include <map>      // std::map
#include <tuple>    // std::make_tuple, std::tuple
#include <vector>   // std::vector

class KokkosCSVM : public ::testing::Test,
                   private util::redirect_output<> { };

TEST_F(KokkosCSVM, construct_parameter) {  // execution_space automatic, target_platform automatic
    // check whether the execution space would be automatically determined as either OpenMPTarget or OpenACC
    const std::map<plssvm::target_platform, std::vector<plssvm::kokkos::execution_space>> available_combinations = plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping();
    plssvm::kokkos::execution_space space{};
    for (const plssvm::target_platform target : plssvm::list_available_target_platforms()) {
        if (plssvm::detail::contains(available_combinations, target)) {
            space = available_combinations.at(target).front();
            break;
        }
    }

    // must throw an exception if the execution space would be OpenMPTarget or OpenACC
    if (space == plssvm::kokkos::execution_space::openmp_target || space == plssvm::kokkos::execution_space::openacc) {
        EXPECT_THROW_WHAT(plssvm::kokkos::csvm{ plssvm::parameter{} },
                          plssvm::kokkos::backend_exception,
                          fmt::format("The Kokkos execution space {} is currently not supported !", space));
    } else {
        EXPECT_NO_THROW(plssvm::kokkos::csvm{ plssvm::parameter{} });
    }
}

TEST_F(KokkosCSVM, construct_target_and_parameter) {  // execution_space automatic, target_platform explicit
    // create parameter struct
    const plssvm::parameter params{};

    // automatic should always work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::automatic, params }));

    const std::map<plssvm::target_platform, std::vector<plssvm::kokkos::execution_space>> available_combinations = plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping();
    const auto target_supported = [&](const plssvm::target_platform target) {
        return plssvm::detail::contains(available_combinations, target);
    };

#if defined(PLSSVM_HAS_CPU_TARGET)
    if (target_supported(plssvm::target_platform::cpu)) {
        EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, params }));
    } else {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, params }),
                          plssvm::kokkos::backend_exception,
                          fmt::format("No Kokkos::ExecutionSpace available ({}) for that requested target platform cpu!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
    }
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, params }),
                      plssvm::kokkos::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    if (target_supported(plssvm::target_platform::gpu_nvidia)) {
        EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, params }));
    } else {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, params }),
                          plssvm::kokkos::backend_exception,
                          fmt::format("No Kokkos::ExecutionSpace available ({}) for that requested target platform gpu_nvidia!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
    }
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, params }),
                      plssvm::kokkos::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

#if defined(PLSSVM_HAS_AMD_TARGET)
    if (target_supported(plssvm::target_platform::gpu_amd)) {
        EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, params }));
    } else {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, params }),
                          plssvm::kokkos::backend_exception,
                          fmt::format("No Kokkos::ExecutionSpace available ({}) for that requested target platform gpu_amd!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
    }
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, params }),
                      plssvm::kokkos::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

#if defined(PLSSVM_HAS_INTEL_TARGET)
    if (target_supported(plssvm::target_platform::gpu_intel)) {
        EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, params }));
    } else {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, params }),
                          plssvm::kokkos::backend_exception,
                          fmt::format("No Kokkos::ExecutionSpace available ({}) for that requested target platform gpu_intel!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
    }
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, params }),
                      plssvm::kokkos::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TEST_F(KokkosCSVM, construct_execution_space_and_parameter) {  // execution_space explicit, target_platform automatic
    // create parameter struct
    const plssvm::parameter params{};

    // automatic should always work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::automatic }));

#if defined(KOKKOS_ENABLE_CUDA)
    // explicitly providing the Cuda execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::cuda }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::cuda }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace Cuda is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_HIP)
    // explicitly providing the HIP execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::hip }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::hip }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace HIP is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_SYCL)
    // explicitly providing the SYCL execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::sycl }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::sycl }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace SYCL is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_HPX)
    // explicitly providing the HPX execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::hpx }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::hpx }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace HPX is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
    // explicitly providing the OpenMP execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openmp }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openmp }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace OpenMP is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    // explicitly providing the OpenMPTarget execution space currently unsupported
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openmp_target }),
                      plssvm::kokkos::backend_exception,
                      "The Kokkos execution space OpenMPTarget is currently not supported !");
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openmp_target }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace OpenMPTarget is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_OPENACC)
    // explicitly providing the OpenACC execution space currently unsupported
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openacc }),
                      plssvm::kokkos::backend_exception,
                      "The Kokkos execution space OpenACC is currently not supported !");
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openacc }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace OpenACC is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_THREADS)
    // explicitly providing the Threads execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::threads }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::threads }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace Threads is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
    // explicitly providing the Serial execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::serial }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ params, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::serial }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace Serial is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif
}

TEST_F(KokkosCSVM, construct_target_and_execution_space_and_parameter) {  // execution_space explicit, target_platform explicit
    // create parameter struct
    const plssvm::parameter params{};

    // list all possible execution spaces
    std::vector<plssvm::kokkos::execution_space> all_execution_spaces{
        plssvm::kokkos::execution_space::cuda,
        plssvm::kokkos::execution_space::hip,
        plssvm::kokkos::execution_space::sycl,
        plssvm::kokkos::execution_space::hpx,
        plssvm::kokkos::execution_space::openmp,
        plssvm::kokkos::execution_space::openmp_target,
        plssvm::kokkos::execution_space::openacc,
        plssvm::kokkos::execution_space::threads,
        plssvm::kokkos::execution_space::serial
    };
    const std::map<plssvm::target_platform, std::vector<plssvm::kokkos::execution_space>> available_combinations = plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping();
    const auto combination_exists = [&](const plssvm::target_platform target, const plssvm::kokkos::execution_space space) {
        return plssvm::detail::contains(available_combinations, target) && plssvm::detail::contains(available_combinations.at(target), space);
    };
    const auto execution_space_available = [&](const plssvm::kokkos::execution_space space) {
        return plssvm::detail::contains(plssvm::kokkos::list_available_execution_spaces(), space);
    };

#if defined(PLSSVM_HAS_CPU_TARGET)
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        if (!execution_space_available(space)) {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, params, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} is not available, available are: {}!", space, fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
        } else if (combination_exists(plssvm::target_platform::cpu, space)) {
            EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, params, plssvm::kokkos_execution_space = space }));
        } else {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, params, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} does not support the requested target platform cpu!", space));
        }
    }
#else
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, params, plssvm::kokkos_execution_space = space }),
                          plssvm::kokkos::backend_exception,
                          "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    }
#endif

#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        if (!execution_space_available(space)) {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, params, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} is not available, available are: {}!", space, fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
        } else if (combination_exists(plssvm::target_platform::gpu_nvidia, space)) {
            EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, params, plssvm::kokkos_execution_space = space }));
        } else {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, params, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} does not support the requested target platform gpu_nvidia!", space));
        }
    }
#else
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, params, plssvm::kokkos_execution_space = space }),
                          plssvm::kokkos::backend_exception,
                          "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    }
#endif

#if defined(PLSSVM_HAS_AMD_TARGET)
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        if (!execution_space_available(space)) {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, params, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} is not available, available are: {}!", space, fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
        } else if (combination_exists(plssvm::target_platform::gpu_amd, space)) {
            EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, params, plssvm::kokkos_execution_space = space }));
        } else {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, params, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} does not support the requested target platform gpu_amd!", space));
        }
    }
#else
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, params, plssvm::kokkos_execution_space = space }),
                          plssvm::kokkos::backend_exception,
                          "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    }
#endif

#if defined(PLSSVM_HAS_INTEL_TARGET)
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        if (!execution_space_available(space)) {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, params, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} is not available, available are: {}!", space, fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
        } else if (combination_exists(plssvm::target_platform::gpu_intel, space)) {
            EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, params, plssvm::kokkos_execution_space = space }));
        } else {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, params, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} does not support the requested target platform gpu_intel!", space));
        }
    }
#else
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, params, plssvm::kokkos_execution_space = space }),
                          plssvm::kokkos::backend_exception,
                          "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    }
#endif
}

TEST_F(KokkosCSVM, construct_named_args) {  // execution_space automatic, target_platform automatic
    // check whether the execution space would be automatically determined as either OpenMPTarget or OpenACC
    const std::map<plssvm::target_platform, std::vector<plssvm::kokkos::execution_space>> available_combinations = plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping();
    plssvm::kokkos::execution_space space{};
    for (const plssvm::target_platform target : plssvm::list_available_target_platforms()) {
        if (plssvm::detail::contains(available_combinations, target)) {
            space = available_combinations.at(target).front();
            break;
        }
    }

    // must throw an exception if the execution space would be OpenMPTarget or OpenACC
    if (space == plssvm::kokkos::execution_space::openmp_target || space == plssvm::kokkos::execution_space::openacc) {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                          plssvm::kokkos::backend_exception,
                          fmt::format("The Kokkos execution space {} is currently not supported !", space));
    } else {
        EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
        EXPECT_NO_THROW(plssvm::kokkos::csvm{ plssvm::cost = 2.0 });
    }
}

TEST_F(KokkosCSVM, construct_target_and_named_args) {  // execution_space automatic, target_platform explicit
    // automatic should always work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::automatic, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));

    const std::map<plssvm::target_platform, std::vector<plssvm::kokkos::execution_space>> available_combinations = plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping();
    const auto target_supported = [&](const plssvm::target_platform target) {
        return plssvm::detail::contains(available_combinations, target);
    };

#if defined(PLSSVM_HAS_CPU_TARGET)
    if (target_supported(plssvm::target_platform::cpu)) {
        EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    } else {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                          plssvm::kokkos::backend_exception,
                          fmt::format("No Kokkos::ExecutionSpace available ({}) for that requested target platform cpu!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
    }
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::kokkos::backend_exception,
                      "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    if (target_supported(plssvm::target_platform::gpu_nvidia)) {
        EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    } else {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                          plssvm::kokkos::backend_exception,
                          fmt::format("No Kokkos::ExecutionSpace available ({}) for that requested target platform gpu_nvidia!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
    }
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::kokkos::backend_exception,
                      "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

#if defined(PLSSVM_HAS_AMD_TARGET)
    if (target_supported(plssvm::target_platform::gpu_amd)) {
        EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    } else {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                          plssvm::kokkos::backend_exception,
                          fmt::format("No Kokkos::ExecutionSpace available ({}) for that requested target platform gpu_amd!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
    }
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::kokkos::backend_exception,
                      "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif

#if defined(PLSSVM_HAS_INTEL_TARGET)
    if (target_supported(plssvm::target_platform::gpu_intel)) {
        EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }));
    } else {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                          plssvm::kokkos::backend_exception,
                          fmt::format("No Kokkos::ExecutionSpace available ({}) for that requested target platform gpu_intel!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
    }
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 }),
                      plssvm::kokkos::backend_exception,
                      "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
#endif
}

TEST_F(KokkosCSVM, construct_execution_space_and_named_args) {  // execution_space explicit, target_platform automatic
    // automatic should always work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::automatic }));

#if defined(KOKKOS_ENABLE_CUDA)
    // explicitly providing the Cuda execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::cuda }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::cuda }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace Cuda is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_HIP)
    // explicitly providing the HIP execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::hip }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::hip }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace HIP is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_SYCL)
    // explicitly providing the SYCL execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::sycl }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::sycl }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace SYCL is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_HPX)
    // explicitly providing the HPX execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::hpx }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::hpx }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace HPX is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
    // explicitly providing the OpenMP execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openmp }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openmp }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace OpenMP is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_OPENMPTARGET)
    // explicitly providing the OpenMPTarget execution space currently unsupported
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openmp_target }),
                      plssvm::kokkos::backend_exception,
                      "The Kokkos execution space OpenMPTarget is currently not supported !");
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openmp_target }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace OpenMPTarget is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_OPENACC)
    // explicitly providing the OpenACC execution space currently unsupported
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openacc }),
                      plssvm::kokkos::backend_exception,
                      "The Kokkos execution space OpenACC is currently not supported !");
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::openacc }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace OpenACC is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_THREADS)
    // explicitly providing the Threads execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::threads }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::threads }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace Threads is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif

#if defined(KOKKOS_ENABLE_SERIAL)
    // explicitly providing the Serial execution space should work
    EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::serial }));
#else
    EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = plssvm::kokkos::execution_space::serial }),
                      plssvm::kokkos::backend_exception,
                      fmt::format("The provided Kokkos::ExecutionSpace Serial is not available, available are: {}!", fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
#endif
}

TEST_F(KokkosCSVM, construct_target_and_execution_space_and_named_args) {  // execution_space explicit, target_platform explicit
    // list all possible execution spaces
    std::vector<plssvm::kokkos::execution_space> all_execution_spaces{
        plssvm::kokkos::execution_space::cuda,
        plssvm::kokkos::execution_space::hip,
        plssvm::kokkos::execution_space::sycl,
        plssvm::kokkos::execution_space::hpx,
        plssvm::kokkos::execution_space::openmp,
        plssvm::kokkos::execution_space::openmp_target,
        plssvm::kokkos::execution_space::openacc,
        plssvm::kokkos::execution_space::threads,
        plssvm::kokkos::execution_space::serial
    };
    const std::map<plssvm::target_platform, std::vector<plssvm::kokkos::execution_space>> available_combinations = plssvm::kokkos::detail::available_target_platform_to_execution_space_mapping();
    const auto combination_exists = [&](const plssvm::target_platform target, const plssvm::kokkos::execution_space space) {
        return plssvm::detail::contains(available_combinations, target) && plssvm::detail::contains(available_combinations.at(target), space);
    };
    const auto execution_space_available = [&](const plssvm::kokkos::execution_space space) {
        return plssvm::detail::contains(plssvm::kokkos::list_available_execution_spaces(), space);
    };

#if defined(PLSSVM_HAS_CPU_TARGET)
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        if (!execution_space_available(space)) {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} is not available, available are: {}!", space, fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
        } else if (combination_exists(plssvm::target_platform::cpu, space)) {
            EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }));
        } else {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} does not support the requested target platform cpu!", space));
        }
    }
#else
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::cpu, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                          plssvm::kokkos::backend_exception,
                          "Requested target platform 'cpu' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    }
#endif

#if defined(PLSSVM_HAS_NVIDIA_TARGET)
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        if (!execution_space_available(space)) {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} is not available, available are: {}!", space, fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
        } else if (combination_exists(plssvm::target_platform::gpu_nvidia, space)) {
            EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }));
        } else {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} does not support the requested target platform gpu_nvidia!", space));
        }
    }
#else
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_nvidia, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                          plssvm::kokkos::backend_exception,
                          "Requested target platform 'gpu_nvidia' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    }
#endif

#if defined(PLSSVM_HAS_AMD_TARGET)
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        if (!execution_space_available(space)) {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} is not available, available are: {}!", space, fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
        } else if (combination_exists(plssvm::target_platform::gpu_amd, space)) {
            EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }));
        } else {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} does not support the requested target platform gpu_amd!", space));
        }
    }
#else
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_amd, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                          plssvm::kokkos::backend_exception,
                          "Requested target platform 'gpu_amd' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    }
#endif

#if defined(PLSSVM_HAS_INTEL_TARGET)
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        if (!execution_space_available(space)) {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} is not available, available are: {}!", space, fmt::join(plssvm::kokkos::list_available_execution_spaces(), ", ")));
        } else if (combination_exists(plssvm::target_platform::gpu_intel, space)) {
            EXPECT_NO_THROW((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }));
        } else {
            EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                              plssvm::kokkos::backend_exception,
                              fmt::format("The provided Kokkos::ExecutionSpace {} does not support the requested target platform gpu_intel!", space));
        }
    }
#else
    for (const plssvm::kokkos::execution_space space : all_execution_spaces) {
        EXPECT_THROW_WHAT((plssvm::kokkos::csvm{ plssvm::target_platform::gpu_intel, plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0, plssvm::kokkos_execution_space = space }),
                          plssvm::kokkos::backend_exception,
                          "Requested target platform 'gpu_intel' that hasn't been enabled using PLSSVM_TARGET_PLATFORMS!");
    }
#endif
}

TEST_F(KokkosCSVM, get_execution_space) {
    // construct default CSVM
    const plssvm::kokkos::csvm svm{ plssvm::parameter{} };

    // after construction: get_execution_space must refer to a plssvm::kokkos::execution_space that is not automatic
    EXPECT_NE(svm.get_execution_space(), plssvm::kokkos::execution_space::automatic);
}

template <bool mock_grid_size, plssvm::kokkos::execution_space space>
struct kokkos_csvm_test_type {
    using mock_csvm_type = mock_kokkos_csvm<mock_grid_size>;
    using csvm_type = plssvm::kokkos::csvm;
    using device_ptr_type = typename csvm_type::device_ptr_type;
    inline static auto additional_arguments = std::make_tuple(std::make_pair(plssvm::kokkos_execution_space, space));
};

namespace impl {

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
template <bool, typename>
struct create_device_tuple_type_helper;

/**
 * @brief Helper struct to create a `std::tuple` containing all available `kokkos_csvm_test_type` types by iterating over the `std::array` of
 *        `plssvm::kokkos::execution_space` values as returned by `plssvm::kokkos::detail::constexpr_available_execution_spaces()`.
 * @tparam mock_grid_size whether the maximum grid size should be mocked (i.e. in fact reduced) or not
 * @tparam Is the indices to index the `std::array`
 */
template <bool mock_grid_size, std::size_t... Is>
struct create_device_tuple_type_helper<mock_grid_size, std::index_sequence<Is...>> {
    /// The array containing all available execution spaces.
    constexpr static auto array = constexpr_available_execution_spaces_to_test();
    /// The resulting variant type.
    using type = std::tuple<kokkos_csvm_test_type<false, array[Is]>...>;
};

/**
 * @brief Create a `std::tuple` containing all available `kokkos_csvm_test_type` types by iterating over the `std::array` of
 *        `plssvm::kokkos::execution_space` values as returned by `plssvm::kokkos::detail::constexpr_available_execution_spaces()`.
 * @tparam mock_grid_size whether the maximum grid size should be mocked (i.e. in fact reduced) or not
 */
template <bool mock_grid_size>
struct create_device_tuple_type {
    /// The number of types in the final variant.
    constexpr static std::size_t N = constexpr_available_execution_spaces_to_test().size();
    /// The final variant type.
    using type = typename create_device_tuple_type_helper<mock_grid_size, std::make_index_sequence<N>>::type;
};

}  // namespace impl

using kokkos_csvm_test_tuple = typename impl::create_device_tuple_type<false>::type;
using kokkos_csvm_test_label_type_list = util::cartesian_type_product_t<kokkos_csvm_test_tuple, plssvm::detail::supported_label_types>;
using kokkos_csvm_test_type_list = util::cartesian_type_product_t<kokkos_csvm_test_tuple>;

// the tests used in the instantiated GTest test suites
using kokkos_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<kokkos_csvm_test_type_list>;
using kokkos_solver_type_gtest = util::combine_test_parameters_gtest_t<kokkos_csvm_test_type_list, util::solver_type_list>;
using kokkos_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<kokkos_csvm_test_type_list, util::kernel_function_type_list>;
using kokkos_solver_and_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<kokkos_csvm_test_type_list, util::solver_and_kernel_function_type_list>;
using kokkos_label_type_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<kokkos_csvm_test_label_type_list, util::kernel_function_and_classification_type_list>;
using kokkos_label_type_solver_kernel_function_and_classification_type_gtest = util::combine_test_parameters_gtest_t<kokkos_csvm_test_label_type_list, util::solver_and_kernel_function_and_classification_type_list>;

// instantiate type-parameterized tests
// generic CSVM tests
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVM, GenericCSVM, kokkos_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVM, GenericCSVMKernelFunction, kokkos_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVM, GenericCSVMSolver, kokkos_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVM, GenericCSVMSolverKernelFunction, kokkos_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVM, GenericCSVMKernelFunctionClassification, kokkos_label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
#if !defined(KOKKOS_ENABLE_CUDA)
// testcase doesn't compile with Kokkos::Cuda's nvcc due to template instantiation limits
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVM, GenericCSVMSolverKernelFunctionClassification, kokkos_label_type_solver_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);
#endif

// generic CSVM DeathTests
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVMDeathTest, GenericCSVMDeathTest, kokkos_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVMDeathTest, GenericCSVMSolverDeathTest, kokkos_solver_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVMDeathTest, GenericCSVMKernelFunctionDeathTest, kokkos_kernel_function_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVMDeathTest, GenericCSVMSolverKernelFunctionDeathTest, kokkos_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU CSVM tests - correct grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVM, GenericGPUCSVM, kokkos_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVM, GenericGPUCSVMKernelFunction, kokkos_kernel_function_type_gtest, naming::test_parameter_to_name);

// generic GPU CSVM DeathTests - correct grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVMDeathTest, GenericGPUCSVMDeathTest, kokkos_csvm_test_type_gtest, naming::test_parameter_to_name);

using kokkos_mock_csvm_test_tuple = typename impl::create_device_tuple_type<true>::type;
using kokkos_mock_csvm_test_type_list = util::cartesian_type_product_t<kokkos_mock_csvm_test_tuple>;

using kokkos_mock_csvm_test_type_gtest = util::combine_test_parameters_gtest_t<kokkos_mock_csvm_test_type_list>;
using kokkos_mock_kernel_function_type_gtest = util::combine_test_parameters_gtest_t<kokkos_mock_csvm_test_type_list, util::kernel_function_type_list>;

// generic GPU CSVM tests - mocked grid sizes
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVMFakedGridSize, GenericGPUCSVM, kokkos_mock_csvm_test_type_gtest, naming::test_parameter_to_name);
INSTANTIATE_TYPED_TEST_SUITE_P(KokkosCSVMFakedGridSize, GenericGPUCSVMKernelFunction, kokkos_mock_kernel_function_type_gtest, naming::test_parameter_to_name);
