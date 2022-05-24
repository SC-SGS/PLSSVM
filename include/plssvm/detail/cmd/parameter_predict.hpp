/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Implements a class encapsulating all necessary parameters for predicting using the C-SVM possibly provided through command line arguments.
*/

#pragma once

#include "plssvm/parameter.hpp"  // plssvm::parameter

#include <string>  // std::string

namespace plssvm::detail::cmd {

/**
* @brief Class for encapsulating all necessary parameters for predicting possibly provided through command line arguments.
*/
class parameter_predict {
 public:
   /**
    * @brief Default construct all parameters for prediction.
    */
   parameter_predict() = default;

   /**
    * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the predict parameters accordingly. Parse the given model and test file.
    * @param[in] argc the number of passed command line arguments
    * @param[in] argv the command line arguments
    */
   parameter_predict(int argc, char **argv);

   backend_type backend = backend_type::automatic;
   /// The target platform: automatic (depending on the used backend), CPUs or GPUs from NVIDIA, AMD or Intel.
   target_platform target = target_platform::automatic;

   /// The SYCL implementation to use with --backend=sycl.
   sycl::implementation_type sycl_implementation_type = sycl::implementation_type::automatic;

   // TODO: here?!?
   /// use strings as label type?
   bool strings_as_labels{ false };
   bool float_as_real_type{ false };

   /// The name of the data/test file to parse.
   std::string input_filename{};
   /// The name of the model file to write the learned support vectors to/to parse the saved model from.
   std::string model_filename{};
   /// The name of the file to write the prediction to.
   std::string predict_filename{};
};

std::ostream &operator<<(std::ostream &out, const parameter_predict &params);

}  // namespace plssvm
