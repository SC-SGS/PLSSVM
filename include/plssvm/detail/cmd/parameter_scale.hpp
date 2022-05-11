/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Implements a class encapsulating all necessary parameters for scaling a data set possibly provided through command line arguments.
*/

#pragma once

#include "plssvm/file_format_types.hpp"  // plssvm::file_format_type

#include <string>  // std::string

namespace plssvm::detail::cmd {

/**
* @brief Class for encapsulating all necessary parameters for scaling a data set possibly provided through command line arguments.
* @tparam T the type of the data
*/
template <typename T>
class parameter_scale {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

 public:
   using real_type = T;

   /**
    * @brief Default construct all training parameters.
    */
   parameter_scale() = default;

   /**
    * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the scale parameters accordingly.
    * @param[in] argc the number of passed command line arguments
    * @param[in] argv the command line arguments
    */
   parameter_scale(int argc, char **argv);

   // TODO: more LIBSVM conform?
   real_type lower{ -1 };
   real_type upper{ +1 };
   file_format_type format{ file_format_type::libsvm };

   /// The name of the data/test file to parse.
   std::string input_filename{};
   /// The name of the model file to write the learned support vectors to/to parse the saved model from.
   std::string scaled_filename{};
};

extern template class parameter_scale<float>;
extern template class parameter_scale<double>;

template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter_scale<T> &params);

}  // namespace plssvm
