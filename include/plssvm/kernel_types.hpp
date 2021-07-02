#pragma once

#include <fmt/format.h>
#include <string_view>

namespace plssvm {

// define possible kernel types
enum class kernel_type {
    linear = 0,
    polynomial = 1,  // (gamma*u'*v + coef0)^degree
    rbf = 2          // exp(-gamma*|u-v|^2)
};

}  // namespace plssvm

template <>
struct fmt::formatter<plssvm::kernel_type> : fmt::formatter<std::string_view> {
    // parse is inherited from formatter<string_view>.
    template <typename FormatContext>
    auto format(plssvm::kernel_type k, FormatContext &ctx) {
        string_view name = "unknown";
        switch (k) {
            case plssvm::kernel_type::linear:
                name = "linear";
                break;
            case plssvm::kernel_type::polynomial:
                name = "polynomial";
                break;
            case plssvm::kernel_type::rbf:
                name = "rbf";
                break;
        }
        return formatter<string_view>::format(name, ctx);
    }
};