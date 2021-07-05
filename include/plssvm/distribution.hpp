#pragma once

#include <plssvm/exceptions/exceptions.hpp>
#include <plssvm/typedef.hpp>

#include <fmt/core.h>

#include <functional>
#include <numeric>
#include <vector>

namespace plssvm {

class distribution {
  public:
    distribution() = default;

    distribution(const std::size_t number_devices, const std::size_t number_lines, const std::vector<real_t> &parameter) {
        if (number_devices != parameter.size()) {
            throw distribution_exception{ fmt::format("Number of devices and parameters mismatch!: {} != {}", number_devices, parameter.size()) };
        }
        const real_t sum = std::accumulate(parameter.cbegin(), parameter.cend(), real_t{ 0.0 });
        for (const real_t value : parameter) {
            distr.push_back(static_cast<int>(value / sum * static_cast<real_t>(number_lines)));
        }
        distr[number_devices - 1] += std::accumulate(distr.cbegin(), distr.cend(), static_cast<int>(number_lines), std::minus<>());
    }

    // TODO: optional implement
    // distribution(const std::size_t number_devices, const std::size_t number_lines) {
    //
    // }

    [[nodiscard]] static bool isValid(const std::size_t number_devices, const std::vector<real_t> &parameter) {
        return number_devices <= parameter.size() && std::accumulate(parameter.cbegin(), parameter.cend(), real_t{ 0.0 }) <= 1.0;
    }

    std::vector<int> distr;
};

}  // namespace plssvm