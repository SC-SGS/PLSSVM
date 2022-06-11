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

#include <iosfwd>  // forward declare std::ostream
#include <string>  // std::string

namespace plssvm::detail::cmd {

/**
 * @brief Class for encapsulating all necessary parameters for scaling a data set possibly provided through command line arguments.
 */
class parameter_scale {
 public:
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

   /// The lower bound of the scaled data values.
   double lower{ -1 };
   /// The upper bound of the scaled data values.
   double upper{ +1 };
   /// The file type (either LIBSVM or ARFF) to which the scaled data should be written to.
   file_format_type format{ file_format_type::libsvm };

   /// `true`if `std::string` should be used as label type instead of the default type `ìnt`.
   bool strings_as_labels{ false };
   /// `true`if `float` should be used as real type instead of the default type `double`.
   bool float_as_real_type{ false };

   /// The name of the data/test file to parse.
   std::string input_filename{};
   /// The name of the model file to write the learned support vectors to/to parse the saved model from.
   std::string scaled_filename{};
   std::string save_filename{};
   std::string restore_filename{};
};

/**
 * @brief Output all scale parameters encapsulated by @p params to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the parameters to
 * @param[in] params the parameters
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const parameter_scale &params);

}  // namespace plssvm
