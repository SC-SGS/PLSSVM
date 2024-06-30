/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the hardware sampler base class.
 */

#ifndef PLSSVM_TESTS_DETAIL_TRACKING_MOCK_HARDWARE_SAMPLER_HPP_
#define PLSSVM_TESTS_DETAIL_TRACKING_MOCK_HARDWARE_SAMPLER_HPP_
#pragma once

#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/target_platforms.hpp"                  //plssvm::target_platform

#include "fmt/core.h"  // fmt::format

#include <chrono>   // std::chrono::steady_clock::time_point
#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <utility>  // std::forward

/**
 * @brief GTest mock class for the base hardware sampler class.
 */
class mock_hardware_sampler final : public plssvm::detail::tracking::hardware_sampler {
  public:
    template <typename... Args>
    explicit mock_hardware_sampler(const std::size_t device_id, Args &&...args) :
        plssvm::detail::tracking::hardware_sampler{ std::forward<Args>(args)... },
        device_id_{ device_id } {
    }

    // basic implementation for pure virtual functions
    [[nodiscard]] std::string generate_yaml_string(std::chrono::steady_clock::time_point) const override {
        return "YAML_string";
    }

    [[nodiscard]] std::string device_identification() const override {
        return fmt::format("device_{}", device_id_);
    }

    [[nodiscard]] plssvm::target_platform sampling_target() const override {
        return plssvm::target_platform::cpu;
    }

  private:
    void sampling_loop() override { }

    std::size_t device_id_{ 0 };
};

#endif  // PLSSVM_TESTS_DETAIL_TRACKING_MOCK_HARDWARE_SAMPLER_HPP_
