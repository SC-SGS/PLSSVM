/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/stdpar/detail/utility.hpp"

#include <string>  // std::string

namespace plssvm::stdpar::detail {

std::string get_stdpar_implementation() {
#if defined(PLSSVM_STDPAR_BACKEND_IMPLEMENTATION)
    return PLSSVM_STDPAR_BACKEND_IMPLEMENTATION;
#else
    return "unknown";
#endif
}

}  // namespace plssvm::stdpar::detail
