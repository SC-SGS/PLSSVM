/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/parameter.hpp"

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name

#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <ostream>  // std::ostream

namespace plssvm::detail {

// explicitly instantiate template class
template struct parameter<float>;
template struct parameter<double>;

template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params) {
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
               detail::arithmetic_type_name<typename parameter<T>::real_type>());
}
template std::ostream &operator<<(std::ostream &, const parameter<float> &);
template std::ostream &operator<<(std::ostream &, const parameter<double> &);

}  // namespace plssvm::detail
