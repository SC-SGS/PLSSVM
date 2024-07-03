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

#ifndef PLSSVM_DETAIL_TRACKING_CPU_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_CPU_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/tracking/cpu/cpu_samples.hpp"   // plssvm::detail::tracking::{cpu_general_samples, clock_samples, power_samples, memory_samples, temperature_samples, gfx_samples, idle_state_samples}
#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/target_platforms.hpp"                  // plssvm::target_platform

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <chrono>  // std::chrono::{steady_clock, milliseconds}, std::chrono_literals namespace
#include <iosfwd>  // std::ostream forward declaration
#include <string>  // std::string

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

/**
 * @brief A hardware sampler for the CPU.
 * @details If available uses the linux commands `turbostat`, `lscpu`, and `free` to gather its information.
 */
class cpu_hardware_sampler : public hardware_sampler {
  public:
    /**
     * @brief Construct a new CPU hardware sampler with the @p sampling_interval.
     * @param[in] sampling_interval the used sampling interval
     */
    explicit cpu_hardware_sampler(std::chrono::milliseconds sampling_interval = PLSSVM_HARDWARE_SAMPLING_INTERVAL);

    /**
     * @brief Delete the copy-constructor (already implicitly deleted due to the base class's std::atomic member).
     */
    cpu_hardware_sampler(const cpu_hardware_sampler &) = delete;
    /**
     * @brief Delete the move-constructor (already implicitly deleted due to the base class's std::atomic member).
     */
    cpu_hardware_sampler(cpu_hardware_sampler &&) noexcept = delete;
    /**
     * @brief Delete the copy-assignment operator (already implicitly deleted due to the base class's std::atomic member).
     */
    cpu_hardware_sampler &operator=(const cpu_hardware_sampler &) = delete;
    /**
     * @brief Delete the move-assignment operator (already implicitly deleted due to the base class's std::atomic member).
     */
    cpu_hardware_sampler &operator=(cpu_hardware_sampler &&) noexcept = delete;

    /**
     * @brief Destruct the CPU hardware sampler. If the sampler is still running, stops it.
     */
    ~cpu_hardware_sampler() override;

    /**
     * @brief Return the general CPU samples of this hardware sampler.
     * @return the general CPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const cpu_general_samples &general_samples() const noexcept { return general_samples_; }

    /**
     * @brief Return the clock related CPU samples of this hardware sampler.
     * @return the clock related CPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const cpu_clock_samples &clock_samples() const noexcept { return clock_samples_; }

    /**
     * @brief Return the power related CPU samples of this hardware sampler.
     * @return the power related CPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const cpu_power_samples &power_samples() const noexcept { return power_samples_; }

    /**
     * @brief Return the memory related CPU samples of this hardware sampler.
     * @return the memory related CPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const cpu_memory_samples &memory_samples() const noexcept { return memory_samples_; }

    /**
     * @brief Return the temperature related CPU samples of this hardware sampler.
     * @return the temperature related CPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const cpu_temperature_samples &temperature_samples() const noexcept { return temperature_samples_; }

    /**
     * @brief Return the gfx (iGPU) related CPU samples of this hardware sampler.
     * @return the gfx (iGPU) related CPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const cpu_gfx_samples &gfx_samples() const noexcept { return gfx_samples_; }

    /**
     * @brief Return the idle state related CPU samples of this hardware sampler.
     * @return the idle state related CPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const cpu_idle_states_samples &idle_state_samples() const noexcept { return idle_state_samples_; }

    /**
     * @copydoc plssvm::detail::tracking::hardware_sampler::generate_yaml_string
     */
    [[nodiscard]] std::string generate_yaml_string(std::chrono::steady_clock::time_point start_time_point) const override;
    /**
     * @copydoc plssvm::detail::tracking::hardware_sampler::device_identification
     */
    [[nodiscard]] std::string device_identification() const override;
    /**
     * @copydoc plssvm::detail::tracking::hardware_sampler::sampling_target
     */
    [[nodiscard]] target_platform sampling_target() const override;

  private:
    /**
     * @copydoc plssvm::detail::tracking::hardware_sampler::sampling_loop
     */
    void sampling_loop() final;

    /// The general CPU samples.
    cpu_general_samples general_samples_{};
    /// The clock related CPU samples.
    cpu_clock_samples clock_samples_{};
    /// The power related CPU samples.
    cpu_power_samples power_samples_{};
    /// The memory related CPU samples.
    cpu_memory_samples memory_samples_{};
    /// The temperature related CPU samples.
    cpu_temperature_samples temperature_samples_{};
    /// The gfx (iGPU) related CPU samples.
    cpu_gfx_samples gfx_samples_{};
    /// The idle state related CPU samples.
    cpu_idle_states_samples idle_state_samples_{};
};

/**
 * @brief Output all CPU samples gathered by the @p sampler to the given output-stream @p out.
 * @details Sets `std::ios_base::failbit` if the @p sampler is still sampling.
 * @param[in,out] out the output-stream to write the CPU samples to
 * @param[in] sampler the CPU hardware sampler
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const cpu_hardware_sampler &sampler);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::cpu_hardware_sampler> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_CPU_HARDWARE_SAMPLER_HPP_
