/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/parameter.hpp"

#include "plssvm/detail/arithmetic_type_name.hpp"  // plssvm::detail::arithmetic_type_name

#include "fmt/core.h"  // fmt::print, fmt::format

#include <string>  // std::string

namespace plssvm {

template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params) {
    return out << fmt::format(
               "kernel_type      {}\n"
               "degree           {}\n"
               "gamma            {}\n"
               "coef0            {}\n"
               "cost             {}\n"
               "epsilon          {}\n"
               "print_info       {}\n"
               "backend          {}\n"
               "target platform  {}\n"
               "input_filename   {}\n"
               "model_filename   {}\n"
               "predict_filename {}\n"
               "real_type        {}\n",
               params.kernel,
               params.degree,
               params.gamma,
               params.coef0,
               params.cost,
               params.epsilon,
               params.print_info,
               params.backend,
               params.target,
               params.input_filename,
               params.model_filename,
               params.predict_filename,
               detail::arithmetic_type_name<typename parameter<T>::real_type>());
}
template std::ostream &operator<<(std::ostream &, const parameter<float> &);
template std::ostream &operator<<(std::ostream &, const parameter<double> &);

// explicitly instantiate template class
template class parameter<float>;
template class parameter<double>;

}  // namespace plssvm
