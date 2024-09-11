/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-train` executable used for training a C-SVM model.
 */

#include "plssvm/core.hpp"
#include "plssvm/detail/cmd/data_set_variants.hpp"         // plssvm::detail::cmd::data_set_factory
#include "plssvm/detail/cmd/parser_train.hpp"              // plssvm::detail::cmd::parser_train
#include "plssvm/detail/logging.hpp"                       // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE,
                                                           // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HARDWARE_SAMPLER_ENTRY, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME
#include "plssvm/detail/utility.hpp"                       // PLSSVM_IS_DEFINED

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
    #include "plssvm/detail/tracking/cpu/hardware_sampler.hpp"      // plssvm::detail::tracking::cpu_hardware_sampler
    #include "plssvm/detail/tracking/hardware_sampler.hpp"          // plssvm::detail::tracking::hardware_sampler
    #include "plssvm/detail/tracking/hardware_sampler_factory.hpp"  // plssvm::detail::tracking::make_hardware_sampler
#endif

#include <algorithm>    // std::for_each
#include <chrono>       // std::chrono::{steady_clock, duration, milliseconds}, std::chrono_literals namespace
#include <cstddef>      // std::size_t
#include <cstdlib>      // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>    // std::exception
#include <functional>   // std::mem_fn
#include <iostream>     // std::cerr, std::endl
#include <memory>       // std::unique_ptr
#include <type_traits>  // std::remove_reference_t
#include <utility>      // std::pair
#include <variant>      // std::visit
#include <vector>       // std::vector

using namespace std::chrono_literals;

int main(int argc, char *argv[]) {
    try {
        const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SET_REFERENCE_TIME(start_time);

        // create and start CPU hardware sampler if available
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
        plssvm::detail::tracking::cpu_hardware_sampler cpu_sampler{ PLSSVM_HARDWARE_SAMPLING_INTERVAL };
        cpu_sampler.start_sampling();
#endif

        // parse SVM parameter from command line
        plssvm::detail::cmd::parser_train cmd_parser{ argc, argv };

        // send warning if the build type is release and assertions are enabled
        if constexpr (std::string_view{ PLSSVM_BUILD_TYPE } == "Release" && PLSSVM_IS_DEFINED(PLSSVM_ENABLE_ASSERTS)) {
            plssvm::detail::log(plssvm::verbosity_level::full | plssvm::verbosity_level::warning,
                                "WARNING: The build type is set to Release, but assertions are enabled. "
                                "This may result in a noticeable performance degradation in parts of PLSSVM!\n");
        }

        // output used parameter
        plssvm::detail::log(plssvm::verbosity_level::full,
                            "\ntask: training\n{}\n\n\n",
                            plssvm::detail::tracking::tracking_entry{ "parameter", "", cmd_parser });

        // create data set
        const auto data_set_visitor = [&](auto &&data) {
            using label_type = typename std::remove_reference_t<decltype(data)>::label_type;

            // check whether SYCL is used as backend (it is either requested directly or as automatic backend)
            const bool use_sycl_as_backend{ cmd_parser.backend == plssvm::backend_type::sycl || (cmd_parser.backend == plssvm::backend_type::automatic && plssvm::determine_default_backend() == plssvm::backend_type::sycl) };

            // create SVM
            const std::unique_ptr<plssvm::csvm> svm = use_sycl_as_backend ? plssvm::make_csvm(cmd_parser.backend, cmd_parser.target, cmd_parser.csvm_params, plssvm::sycl_implementation_type = cmd_parser.sycl_implementation_type, plssvm::sycl_kernel_invocation_type = cmd_parser.sycl_kernel_invocation_type)
                                                                          : plssvm::make_csvm(cmd_parser.backend, cmd_parser.target, cmd_parser.csvm_params);

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
            // initialize hardware sampling
            std::vector<std::unique_ptr<plssvm::detail::tracking::hardware_sampler>> sampler =
                plssvm::detail::tracking::create_hardware_sampler(svm->get_target_platform(), svm->num_available_devices(), PLSSVM_HARDWARE_SAMPLING_INTERVAL);
            // start sampling
            std::for_each(sampler.begin(), sampler.end(), std::mem_fn(&plssvm::detail::tracking::hardware_sampler::start_sampling));
#endif

            // only specify plssvm::max_iter if it isn't its default value
            const plssvm::model<label_type> model =
                cmd_parser.max_iter == std::size_t{ 0 }
                    ? svm->fit(data,
                               plssvm::epsilon = cmd_parser.epsilon,
                               plssvm::classification = cmd_parser.classification,
                               plssvm::solver = cmd_parser.solver)
                    : svm->fit(data,
                               plssvm::epsilon = cmd_parser.epsilon,
                               plssvm::max_iter = cmd_parser.max_iter,
                               plssvm::classification = cmd_parser.classification,
                               plssvm::solver = cmd_parser.solver);
            // save model to file
            model.save(cmd_parser.model_filename);

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
            // stop sampling
            std::for_each(sampler.begin(), sampler.end(), std::mem_fn(&plssvm::detail::tracking::hardware_sampler::stop_sampling));
            // write samples to yaml file
            std::for_each(sampler.cbegin(), sampler.cend(), [&](const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> &s) {
                PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HARDWARE_SAMPLER_ENTRY(*s);
            });
#endif
        };
        std::visit(data_set_visitor, plssvm::detail::cmd::data_set_factory(cmd_parser));

        // stop CPU hardware sampler and dump results if available
#if defined(PLSSVM_HARDWARE_TRACKING_FOR_CPUS_ENABLED)
        cpu_sampler.stop_sampling();
        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_HARDWARE_SAMPLER_ENTRY(cpu_sampler);
#endif

        const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        plssvm::detail::log(plssvm::verbosity_level::full,
                            "\nTotal runtime: {}\n",
                            plssvm::detail::tracking::tracking_entry{ "", "total_time", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time) });

        PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(cmd_parser.performance_tracking_filename);

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
