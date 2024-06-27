/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-scale` executable used for scaling a data set to a specified range.
 */

#include "plssvm/core.hpp"
#include "plssvm/detail/cmd/data_set_variants.hpp"         // plssvm::detail::cmd::data_set_factory
#include "plssvm/detail/cmd/parser_scale.hpp"              // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/logging.hpp"                       // plssvm::detail::log
#include "plssvm/detail/tracking/performance_tracker.hpp"  // plssvm::detail::tracking::tracking_entry, PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE
#include "plssvm/detail/utility.hpp"                       // PLSSVM_IS_DEFINED

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
    #include "plssvm/detail/tracking/hardware_sampler.hpp"          // plssvm::detail::tracking::hardware_sampler
    #include "plssvm/detail/tracking/hardware_sampler_factory.hpp"  // plssvm::detail::tracking::make_hardware_sampler
#endif

#include <algorithm>   // std::for_each
#include <chrono>      // std::chrono::{steady_clock, duration}, std::chrono_literals namespace
#include <cstddef>     // std::size_t
#include <cstdlib>     // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>   // std::exception
#include <functional>  // std::mem_fn
#include <iostream>    // std::cerr, std::endl
#include <utility>     // std::pair
#include <variant>     // std::visit
#include <vector>      // std::vector

using namespace std::chrono_literals;

int main(int argc, char *argv[]) {
    try {
        const std::chrono::steady_clock::time_point start_time = std::chrono::steady_clock::now();

        // create default parameters
        const plssvm::detail::cmd::parser_scale cmd_parser{ argc, argv };

        // send warning if the build type is release and assertions are enabled
        if constexpr (std::string_view{ PLSSVM_BUILD_TYPE } == "Release" && PLSSVM_IS_DEFINED(PLSSVM_ENABLE_ASSERTS)) {
            plssvm::detail::log(plssvm::verbosity_level::full | plssvm::verbosity_level::warning,
                                "WARNING: The build type is set to Release, but assertions are enabled. "
                                "This may result in a noticeable performance degradation in parts of PLSSVM!\n");
        }

        // output used parameter
        plssvm::detail::log(plssvm::verbosity_level::full,
                            "\ntask: scaling\n{}\n",
                            plssvm::detail::tracking::tracking_entry{ "parameter", "", cmd_parser });

        // create data set and scale
        const auto data_set_visitor = [&](auto &&data) {
        // initialize hardware sampling -> plssvm-scale is CPU only!
#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
            // initialize hardware sampling
            std::vector<std::unique_ptr<plssvm::detail::tracking::hardware_sampler>> sampler =
                plssvm::detail::tracking::create_hardware_sampler(plssvm::target_platform::cpu, std::size_t{ 1 });
            // start sampling
            std::for_each(sampler.begin(), sampler.end(), std::mem_fn(&plssvm::detail::tracking::hardware_sampler::start_sampling));
#endif

            // write scaled data to output file
            if (!cmd_parser.scaled_filename.empty()) {
                data.save(cmd_parser.scaled_filename, cmd_parser.format);
            } else {
                fmt::print("\n");
                using label_type = typename plssvm::detail::remove_cvref_t<decltype(data)>::label_type;

                // output to console if no output filename is provided
                const auto &matrix = data.data();
                const plssvm::optional_ref<const std::vector<label_type>> label = data.labels();
                for (std::size_t row = 0; row < matrix.num_rows(); ++row) {
                    if (label.has_value()) {
                        fmt::print(FMT_COMPILE("{} "), label.value().get()[row]);
                    }
                    for (std::size_t col = 0; col < matrix.num_cols(); ++col) {
                        if (matrix(row, col) != plssvm::real_type{ 0.0 }) {
                            fmt::print(FMT_COMPILE("{}:{:.10e} "), col + 1, matrix(row, col));
                        }
                    }
                    fmt::print("\n");
                }
            }

            // save scaling parameters if requested
            if (!cmd_parser.save_filename.empty() && data.scaling_factors().has_value()) {
                data.scaling_factors()->get().save(cmd_parser.save_filename);
            }

#if defined(PLSSVM_HARDWARE_SAMPLING_ENABLED)
            // stop sampling
            std::for_each(sampler.begin(), sampler.end(), std::mem_fn(&plssvm::detail::tracking::hardware_sampler::stop_sampling));
            // write samples to yaml file
            std::for_each(sampler.cbegin(), sampler.cend(), [&](const std::unique_ptr<plssvm::detail::tracking::hardware_sampler> &s) {
                using track_type = std::pair<plssvm::detail::tracking::hardware_sampler *, std::chrono::steady_clock::time_point>;
                PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY((plssvm::detail::tracking::tracking_entry{ "hardware_samples", s->device_identification(), track_type{ s.get(), start_time } }));
            });
#endif
        };
        std::visit(data_set_visitor, plssvm::detail::cmd::data_set_factory(cmd_parser));

        const std::chrono::steady_clock::time_point end_time = std::chrono::steady_clock::now();
        plssvm::detail::log(plssvm::verbosity_level::full | plssvm::verbosity_level::timing,
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
