/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/OpenCL/exceptions.hpp"

#include "plssvm/exceptions/exceptions.hpp"       // plssvm::exception
#include "plssvm/exceptions/source_location.hpp"  // plssvm::source_location

#include <string>  // std::string

namespace plssvm::opencl {

backend_exception::backend_exception(const std::string &msg, source_location loc) :
    ::plssvm::exception{ msg, "opencl::backend_exception", loc } { }

}  // namespace plssvm::opencl
