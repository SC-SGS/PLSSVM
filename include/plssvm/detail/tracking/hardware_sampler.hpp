/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all hardware samplers.
 */

#ifndef PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
#define PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::remove_cvref_t
#include "plssvm/target_platforms.hpp"    // plssvm::target_platform

#include <atomic>  // std::atomic
#include <chrono>  // std::chrono::{steady_clock::time_point, milliseconds}
#include <string>  // std::string
#include <thread>  // std::thread
#include <vector>  // std::vector

namespace plssvm::detail::tracking {

/**
 * @brief The base class for all specialized hardware samplers.
 */
class hardware_sampler {
  public:
    /**
     * @brief Construct a new hardware sampler with the provided @p sampling_interval.
     * @param[in] sampling_interval the used sampling interval
     */
    explicit hardware_sampler(std::chrono::milliseconds sampling_interval);

    /**
     * @brief Delete the copy-constructor (already implicitly deleted due to the std::atomic member).
     */
    hardware_sampler(const hardware_sampler &) = delete;
    /**
     * @brief Delete the move-constructor (already implicitly deleted due to the std::atomic member).
     */
    hardware_sampler(hardware_sampler &&) noexcept = delete;
    /**
     * @brief Delete the copy-assignment operator (already implicitly deleted due to the std::atomic member).
     */
    hardware_sampler &operator=(const hardware_sampler &) = delete;
    /**
     * @brief Delete the move-assignment operator (already implicitly deleted due to the std::atomic member).
     */
    hardware_sampler &operator=(hardware_sampler &&) noexcept = delete;

    /**
     * @brief Pure virtual default destructor.
     */
    virtual ~hardware_sampler() = 0;

    /**
     * @brief Start hardware sampling in a new std::thread.
     * @details Once a hardware sampler has been started, it can never be started again, even if `hardware_sampler::stop_sampling` has been called.
     * @throws plssvm::hardware_sampling_exception if the hardware sampler has already been started
     */
    void start_sampling();
    /**
     * @brief Stop hardware sampling. Signals the running std::thread to stop sampling and joins it.
     * @details Once a hardware sampler has been stopped, it can never be stopped again.
     * @throws plssvm::hardware_sampling_exception if the hardware sampler hasn't been started yet
     * @throws plssvm::hardware_sampling_exception if the hardware sampler has already been stopped
     */
    void stop_sampling();
    /**
     * @brief Pause hardware sampling.
     */
    void pause_sampling();
    /**
     * @brief Resume hardware sampling.
     * @throws plssvm::hardware_sampling_exception if the hardware sampler has already been stopped
     */
    void resume_sampling();

    /**
     * @brief Check whether this hardware sampler has already started sampling.
     * @return `true` if the hardware sampler has already started sampling, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool has_sampling_started() const noexcept;
    /**
     * @brief Check whether this hardware sampler has currently sampling.
     * @return `true` if the hardware sampler is currently sampling, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_sampling() const noexcept;
    /**
     * @brief Check whether this hardware sampler has already stopped sampling.
     * @return `true` if the hardware sampler has already stopped sampling, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool has_sampling_stopped() const noexcept;

    /**
     * @brief Return the time points the samples of this hardware sampler occurred.
     * @return the time points (`[[nodiscard]]`)
     */
    [[nodiscard]] std::vector<std::chrono::steady_clock::time_point> time_points() const noexcept { return time_points_; }

    /**
     * @brief Return the sampling interval of this hardware sampler.
     * @return the samping interval in milliseconds (`[[nodiscard]]`)
     */
    [[nodiscard]] std::chrono::milliseconds sampling_interval() const noexcept { return sampling_interval_; }

    /**
     * @brief Assemble the YAML string containing all hardware samples.
     * @param[in] start_time_point the reference time point the hardware samples occurred relative to
     * @return the YAML string (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::string generate_yaml_string(std::chrono::steady_clock::time_point start_time_point) const = 0;
    /**
     * @brief Return the unique device identification. Can be used as unique key in the YAML string.
     * @return the unique device identification (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual std::string device_identification() const = 0;
    /**
     * @brief Returns the target platform this hardware sampler is responsible for.
     * @return the target platform (`[[nodiscard]]`)
     */
    [[nodiscard]] virtual target_platform sampling_target() const = 0;

  protected:
    /**
     * @brief Getter the hardware samples. Called in another std::thread.
     */
    virtual void sampling_loop() = 0;

    /**
     * @brief Add a new time point to this hardware sampler. Called during the sampling loop.
     * @param time_point the new time point to add
     */
    void add_time_point(const std::chrono::steady_clock::time_point time_point) { time_points_.push_back(time_point); }

  private:
    /// A boolean flag indicating whether the sampling has already started.
    std::atomic<bool> sampling_started_{ false };
    /// A boolean flag indicating whether the sampling has already stopped.
    std::atomic<bool> sampling_stopped_{ false };
    /// A boolean flag indicating whether the sampling is currently running.
    std::atomic<bool> sampling_running_{ false };

    /// The std::thread used to getter the hardware samples.
    std::thread sampling_thread_{};

    /// The time points at which this hardware sampler sampled its values.
    std::vector<std::chrono::steady_clock::time_point> time_points_{};

    /// The sampling interval of this hardware sampler.
    const std::chrono::milliseconds sampling_interval_{};
};

/// @cond Doxygen_suppress
// Forward declare all possible hardware sampler.
class cpu_hardware_sampler;
class gpu_nvidia_hardware_sampler;
class gpu_amd_hardware_sampler;
class gpu_intel_hardware_sampler;

/**
 * @brief No `value` member variable if anything other than a hardware sampler has been provided.
 */
template <typename T>
struct hardware_sampler_to_target_platform_impl { };

/**
 * @brief Sets the `value` to `plssvm::target_platform::cpu` for the CPU hardware sampler.
 */
template <>
struct hardware_sampler_to_target_platform_impl<cpu_hardware_sampler> {
    /// The enum value representing the CPU target.
    constexpr static target_platform value = target_platform::cpu;
};

/**
 * @brief Sets the `value` to `plssvm::target_platform::gpu_nvidia` for the NVIDIA GPU hardware sampler.
 */
template <>
struct hardware_sampler_to_target_platform_impl<gpu_nvidia_hardware_sampler> {
    /// The enum value representing the NVIDIA GPU target.
    constexpr static target_platform value = target_platform::gpu_nvidia;
};

/**
 * @brief Sets the `value` to `plssvm::target_platform::gpu_amd` for the AMD GPU hardware sampler.
 */
template <>
struct hardware_sampler_to_target_platform_impl<gpu_amd_hardware_sampler> {
    /// The enum value representing the AMD GPU target.
    constexpr static target_platform value = target_platform::gpu_amd;
};

/**
 * @brief Sets the `value` to `plssvm::target_platform::gpu_intel` for the Intel GPU hardware sampler.
 */
template <>
struct hardware_sampler_to_target_platform_impl<gpu_intel_hardware_sampler> {
    /// The enum value representing the Intel GPU target.
    constexpr static target_platform value = target_platform::gpu_intel;
};

/// @endcond

/**
 * @brief Get the plssvm::target_platform of the hardware sampler class of type @p T. Ignores all top-level const, volatile, and reference qualifiers.
 * @details Provides a member variable `value` if @p T is a valid hardware sampler.
 * @tparam T the type of the hardware sampler to get the backend type from
 */
template <typename T>
struct hardware_sampler_to_target_platform : hardware_sampler_to_target_platform_impl<detail::remove_cvref_t<T>> { };

/**
 * @copydoc plssvm::detail::tracking::hardware_sampler_to_target_platform
 * @details A shorthand for `plssvm::hardware_sampler_to_target_platform::value`.
 */
template <typename T>
constexpr target_platform hardware_sampler_to_target_platform_v = hardware_sampler_to_target_platform<T>::value;

}  // namespace plssvm::detail::tracking

#endif  // PLSSVM_DETAIL_TRACKING_HARDWARE_SAMPLER_HPP_
