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
    using base_type::print_info;
    using base_type::target;
    using base_type::sycl_kernel_invocation_type;

    using base_type::input_filename;
    using base_type::model_filename;
    using base_type::predict_filename;

    using base_type::alpha_ptr;
    using base_type::data_ptr;
    using base_type::test_data_ptr;
    using base_type::value_ptr;

    using base_type::rho;

    /**
     * @brief Default construct all training parameters.
     */
    parameter_train() = default;

    /**
     * @brief Set all training parameters to their default values and parse the data file.
     * @details Sets the model_filename to `${input_filename}.model`.
     * @param[in] input_filename the name of the data file
     */
    explicit parameter_train(std::string input_filename);

    /**
     * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the training parameters accordingly. Parse the given data file.
     * @param[in] argc the number of passed command line arguments
     * @param[in] argv the command line arguments
     */
    parameter_train(int argc, char **argv);
};

extern template class parameter_train<float>;
extern template class parameter_train<double>;

}  // namespace plssvm
