/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/tracking/gpu_intel/level_zero_samples.hpp"

#include "plssvm/detail/tracking/utility.hpp"  // plssvm::detail::tracking::value_or_default

#include "fmt/core.h"    // fmt::format
#include "fmt/format.h"  // fmt::join

#include <cstddef>   // std::size_t
#include <optional>  // std::optional
#include <ostream>   // std::ostream
#include <string>    // std::string
#include <vector>    // std::vector

namespace plssvm::detail::tracking {

//*************************************************************************************************************************************//
//                                                           general samples                                                           //
//*************************************************************************************************************************************//

std::string level_zero_general_samples::generate_yaml_string() const {
    std::string str{ "    general:\n" };

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const level_zero_general_samples &samples) {
    return out << fmt::format("");
}

//*************************************************************************************************************************************//
//                                                            clock samples                                                            //
//*************************************************************************************************************************************//

std::string level_zero_clock_samples::generate_yaml_string() const {
    std::string str{ "    clock:\n" };

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const level_zero_clock_samples &samples) {
    return out << fmt::format("");
}

//*************************************************************************************************************************************//
//                                                            power samples                                                            //
//*************************************************************************************************************************************//

std::string level_zero_power_samples::generate_yaml_string() const {
    std::string str{ "    power:\n" };

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const level_zero_power_samples &samples) {
    return out << fmt::format("");
}

//*************************************************************************************************************************************//
//                                                            memory samples                                                           //
//*************************************************************************************************************************************//

std::string level_zero_memory_samples::generate_yaml_string() const {
    std::string str{ "    memory:\n" };

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const level_zero_memory_samples &samples) {
    return out << fmt::format("");
}

//*************************************************************************************************************************************//
//                                                         temperature samples                                                         //
//*************************************************************************************************************************************//

std::string level_zero_temperature_samples::generate_yaml_string() const {
    std::string str{ "    temperature:\n" };

    // remove last newline
    str.pop_back();

    return str;
}

std::ostream &operator<<(std::ostream &out, const level_zero_temperature_samples &samples) {
    return out << fmt::format("");
}

}  // namespace plssvm::detail::tracking
