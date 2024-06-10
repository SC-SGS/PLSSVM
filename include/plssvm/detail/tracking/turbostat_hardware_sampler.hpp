/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a hardware sampler for CPUs using the turbostat utility (requires root).
 */

#ifndef PLSSVM_DETAIL_TRACKING_TURBOSTAT_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_TURBOSTAT_HARDWARE_SAMPLER_HPP_

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler

#include <atomic>   // std::atomic
#include <chrono>   // std::chrono::milliseconds, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

class turbostat_hardware_sampler : public hardware_sampler {
  public:
    explicit turbostat_hardware_sampler(std::chrono::milliseconds sampling_interval = 100ms);

    turbostat_hardware_sampler(const turbostat_hardware_sampler &) = delete;
    turbostat_hardware_sampler(turbostat_hardware_sampler &&) noexcept = delete;
    turbostat_hardware_sampler &operator=(const turbostat_hardware_sampler &) = delete;
    turbostat_hardware_sampler &operator=(turbostat_hardware_sampler &&) noexcept = delete;

    ~turbostat_hardware_sampler() override = default;

    [[nodiscard]] std::string device_identification() const noexcept override;

    [[nodiscard]] std::string assemble_yaml_sample_string() const override;

  private:
    void sampling_loop() final;

    std::vector<std::chrono::milliseconds> time_since_start_{};
    std::vector<std::string> data_lines_{};
};

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_TURBOSTAT_HARDWARE_SAMPLER_HPP_
