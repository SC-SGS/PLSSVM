/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/parameter.hpp"

#include "plssvm/constants.hpp"                    // plssvm::real_type
#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <ostream>  // std::ostream

namespace plssvm {

std::ostream &operator<<(std::ostream &out, const parameter &params) {
    return out << fmt::format(
               "kernel_type                 {}\n"
               "degree                      {}\n"
               "gamma                       {}\n"
               "coef0                       {}\n"
               "cost                        {}\n"
               "real_type                   {}\n",
               params.kernel_type,
               params.degree,
               params.gamma,
               params.coef0,
               params.cost,
               detail::arithmetic_type_name<real_type>());
}

}  // namespace plssvm
