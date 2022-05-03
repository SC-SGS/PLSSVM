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

namespace plssvm {

/**
 * @brief Class for encapsulating all necessary parameters for training possibly provided through command line arguments.
 * @tparam T the type of the data
 */
template <typename T>
class parameter_train : public parameter<T> {
  public:
    /// The template base type of the parameter_train class.
    using base_type = parameter<T>;

    using base_type::backend;
    using base_type::coef0;
    using base_type::cost;
    using base_type::degree;
    using base_type::epsilon;
    using base_type::gamma;
    using base_type::kernel;
    using base_type::target;
    using base_type::sycl_kernel_invocation_type;
    using base_type::sycl_implementation_type;

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

    /// The name of the data/test file to parse.
    std::string input_filename{};
    /// The name of the model file to write the learned support vectors to/to parse the saved model from.
    std::string model_filename{};

  private:
    /**
     * @brief Generate a model filename based on the name of the input file.
     * @return `${input_filename}.model` (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string model_name_from_input();
};

extern template class parameter_train<float>;
extern template class parameter_train<double>;

template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter_train<T> &params);

}  // namespace plssvm
