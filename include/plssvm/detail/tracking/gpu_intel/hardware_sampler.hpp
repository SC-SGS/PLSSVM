/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a hardware sampler for Intel GPUs using Intel's Level Zero.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_INTEL_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_INTEL_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/tracking/gpu_intel/level_zero_device_handle.hpp"  // plssvm::detail::tracking::level_zero_device_handle
#include "plssvm/detail/tracking/gpu_intel/level_zero_samples.hpp"        // plssvm::detail::tracking::{level_zero_general_samples, level_zero_clock_samples, level_zero_power_samples, level_zero_memory_samples, level_zero_temperature_samples}
#include "plssvm/detail/tracking/hardware_sampler.hpp"                    // plssvm::detail::tracking::hardware_sampler
#include "plssvm/target_platforms.hpp"                                    // plssvm::target_platform

#include <atomic>   // std::atomic
#include <chrono>   // std::chrono::{steady_clock, milliseconds}, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <string>   // std::string

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

/**
 * @brief A hardware sampler for Intel GPUs.
 * @details Uses Intel's Level Zero library.
 */
class gpu_intel_hardware_sampler : public hardware_sampler {
  public:
    /**
     * @brief Construct a new Intel GPU hardware sampler for device @p device_id with the @p sampling_interval.
     * @details If this is the first Intel GPU sampler, initializes the Level Zero environment.
     * @param[in] device_id the ID of the device to sample
     * @param[in] sampling_interval the used sampling interval
     */
    explicit gpu_intel_hardware_sampler(std::size_t device_id, std::chrono::milliseconds sampling_interval = PLSSVM_HARDWARE_SAMPLING_INTERVAL);

    /**
     * @brief Delete the copy-constructor (already implicitly deleted due to the base class's std::atomic member).
     */
    gpu_intel_hardware_sampler(const gpu_intel_hardware_sampler &) = delete;
    /**
     * @brief Delete the move-constructor (already implicitly deleted due to the base class's std::atomic member).
     */
    gpu_intel_hardware_sampler(gpu_intel_hardware_sampler &&) noexcept = delete;
    /**
     * @brief Delete the copy-assignment operator (already implicitly deleted due to the base class's std::atomic member).
     */
    gpu_intel_hardware_sampler &operator=(const gpu_intel_hardware_sampler &) = delete;
    /**
     * @brief Delete the move-assignment operator (already implicitly deleted due to the base class's std::atomic member).
     */
    gpu_intel_hardware_sampler &operator=(gpu_intel_hardware_sampler &&) noexcept = delete;

    /**
     * @brief Destruct the Intel GPU hardware sampler. If the sampler is still running, stops it.
     */
    ~gpu_intel_hardware_sampler() override;

    /**
     * @brief Return the general Intel GPU samples of this hardware sampler.
     * @return the general Intel GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const level_zero_general_samples &general_samples() const noexcept { return general_samples_; }

    /**
     * @brief Return the clock related Intel GPU samples of this hardware sampler.
     * @return the clock related Intel GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const level_zero_clock_samples &clock_samples() const noexcept { return clock_samples_; }

    /**
     * @brief Return the power related Intel GPU samples of this hardware sampler.
     * @return the power related Intel GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const level_zero_power_samples &power_samples() const noexcept { return power_samples_; }

    /**
     * @brief Return the memory related Intel GPU samples of this hardware sampler.
     * @return the memory related Intel GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const level_zero_memory_samples &memory_samples() const noexcept { return memory_samples_; }

    /**
     * @brief Return the temperature related Intel GPU samples of this hardware sampler.
     * @return the temperature related Intel GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const level_zero_temperature_samples &temperature_samples() const noexcept { return temperature_samples_; }

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

    /// The device handle for the device to sample.
    level_zero_device_handle device_;

    /// The general Intel GPU samples.
    level_zero_general_samples general_samples_{};
    /// The clock related Intel GPU samples.
    level_zero_clock_samples clock_samples_{};
    /// The power related Intel GPU samples.
    level_zero_power_samples power_samples_{};
    /// The memory related Intel GPU samples.
    level_zero_memory_samples memory_samples_{};
    /// The temperature related Intel GPU samples.
    level_zero_temperature_samples temperature_samples_{};

    /// The total number of currently active Intel GPU hardware samplers.
    inline static std::atomic<int> instances_{ 0 };
    /// True if the Level Zero environment has been successfully initialized (only done by a single hardware sampler).
    inline static std::atomic<bool> init_finished_{ false };
};

/**
 * @brief Output all Intel GPU samples gathered by the @p sampler to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the Intel GPU samples to
 * @param[in] sampler the Intel GPU hardware sampler
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const gpu_intel_hardware_sampler &sampler);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::gpu_intel_hardware_sampler> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_GPU_INTEL_HARDWARE_SAMPLER_HPP_
