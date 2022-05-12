/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Implements a class encapsulating all necessary parameters for training the C-SVM possibly provided through command line arguments.
*/

#pragma once

#include "plssvm/parameter.hpp"  // plssvm::parameter

#include <string>  // std::string

namespace plssvm::detail::cmd {

/**
* @brief Class for encapsulating all necessary parameters for training possibly provided through command line arguments.
*/
class parameter_train {
  public:
   /**
    * @brief Default construct all training parameters.
    */
   parameter_train() = default;

   /**
    * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the training parameters accordingly. Parse the given data file.
    * @param[in] argc the number of passed command line arguments
    * @param[in] argv the command line arguments
    */
   parameter_train(int argc, char **argv);

   /// Other parameters
   parameter_variants base_params{};

   /// The name of the data/test file to parse.
   std::string input_filename{};
   /// The name of the model file to write the learned support vectors to/to parse the saved model from.
   std::string model_filename{};
};

std::ostream &operator<<(std::ostream &out, const parameter_train &params);

}  // namespace plssvm
