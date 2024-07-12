/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a hardware sampler for AMD GPUs using AMD's ROCm SMI library.
 */

#ifndef PLSSVM_DETAIL_TRACKING_GPU_AMD_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_GPU_AMD_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/tracking/gpu_amd/rocm_smi_samples.hpp"  // plssvm::detail::tracking::{rocm_smi_general_samples, rocm_smi_clock_samples, rocm_smi_power_samples, rocm_smi_memory_samples, rocm_smi_temperature_samples}
#include "plssvm/detail/tracking/hardware_sampler.hpp"          // plssvm::detail::tracking::hardware_sampler
#include "plssvm/target_platforms.hpp"                          // plssvm::target_platform

#include "fmt/core.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <atomic>   // std::atomic
#include <chrono>   // std::chrono::{steady_clock, milliseconds}, std::chrono_literals namespace
#include <cstddef>  // std::size_t
#include <cstdint>  // std::uint32_t
#include <iosfwd>   // std::ostream forward declaration
#include <string>   // std::string

namespace plssvm::detail::tracking {

using namespace std::chrono_literals;

/**
 * @brief A hardware sampler for AMD GPUs.
 * @details Uses AMD's ROCm SMI library.
 */
class gpu_amd_hardware_sampler : public hardware_sampler {
  public:
    /**
     * @brief Construct a new AMD GPU hardware sampler for device @p device_id with the @p sampling_interval.
     * @details If this is the first AMD GPU sampler, initializes the ROCm SMI environment.
     * @param[in] device_id the ID of the device to sample
     * @param[in] sampling_interval the used sampling interval
     */
    explicit gpu_amd_hardware_sampler(std::size_t device_id, std::chrono::milliseconds sampling_interval = PLSSVM_HARDWARE_SAMPLING_INTERVAL);

    /**
     * @brief Delete the copy-constructor (already implicitly deleted due to the base class's std::atomic member).
     */
    gpu_amd_hardware_sampler(const gpu_amd_hardware_sampler &) = delete;
    /**
     * @brief Delete the move-constructor (already implicitly deleted due to the base class's std::atomic member).
     */
    gpu_amd_hardware_sampler(gpu_amd_hardware_sampler &&) noexcept = delete;
    /**
     * @brief Delete the copy-assignment operator (already implicitly deleted due to the base class's std::atomic member).
     */
    gpu_amd_hardware_sampler &operator=(const gpu_amd_hardware_sampler &) = delete;
    /**
     * @brief Delete the move-assignment operator (already implicitly deleted due to the base class's std::atomic member).
     */
    gpu_amd_hardware_sampler &operator=(gpu_amd_hardware_sampler &&) noexcept = delete;

    /**
     * @brief Destruct the AMD GPU hardware sampler. If the sampler is still running, stops it.
     * @details If this is the last AMD GPU sampler, cleans up the ROCm SMI environment.
     */
    ~gpu_amd_hardware_sampler() override;

    /**
     * @brief Return the general AMD GPU samples of this hardware sampler.
     * @return the general AMD GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const rocm_smi_general_samples &general_samples() const noexcept { return general_samples_; }

    /**
     * @brief Return the clock related AMD GPU samples of this hardware sampler.
     * @return the clock related AMD GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const rocm_smi_clock_samples &clock_samples() const noexcept { return clock_samples_; }

    /**
     * @brief Return the power related AMD GPU samples of this hardware sampler.
     * @return the power related AMD GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const rocm_smi_power_samples &power_samples() const noexcept { return power_samples_; }

    /**
     * @brief Return the memory related AMD GPU samples of this hardware sampler.
     * @return the memory related AMD GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const rocm_smi_memory_samples &memory_samples() const noexcept { return memory_samples_; }

    /**
     * @brief Return the temperature related AMD GPU samples of this hardware sampler.
     * @return the temperature related AMD GPU samples (`[[nodiscard]]`)
     */
    [[nodiscard]] const rocm_smi_temperature_samples &temperature_samples() const noexcept { return temperature_samples_; }

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

    /// The ID of the device to sample.
    std::uint32_t device_id_{};

    /// The general AMD GPU samples.
    rocm_smi_general_samples general_samples_{};
    /// The clock related AMD GPU samples.
    rocm_smi_clock_samples clock_samples_{};
    /// The power related AMD GPU samples.
    rocm_smi_power_samples power_samples_{};
    /// The memory related AMD GPU samples.
    rocm_smi_memory_samples memory_samples_{};
    /// The temperature related AMD GPU samples.
    rocm_smi_temperature_samples temperature_samples_{};

    /// The total number of currently active AMD GPU hardware samplers.
    inline static std::atomic<int> instances_{ 0 };
    /// True if the ROCm SMI environment has been successfully initialized (only done by a single hardware sampler).
    inline static std::atomic<bool> init_finished_{ false };
};

/**
 * @brief Output all AMD GPU samples gathered by the @p sampler to the given output-stream @p out.
 * @details Sets `std::ios_base::failbit` if the @p sampler is still sampling.
 * @param[in,out] out the output-stream to write the AMD GPU samples to
 * @param[in] sampler the AMD GPU hardware sampler
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const gpu_amd_hardware_sampler &sampler);

}  // namespace plssvm::detail::tracking

template <>
struct fmt::formatter<plssvm::detail::tracking::gpu_amd_hardware_sampler> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_GPU_AMD_HARDWARE_SAMPLER_HPP_
