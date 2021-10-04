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

namespace plssvm {

/**
 * @brief Class for encapsulating all necessary parameters for predicting possibly provided through command line arguments.
 * @tparam T the type of the data
 */
template <typename T>
class parameter_predict : public parameter<T> {
  public:
    /// The template base type of the parameter_predict class.
    using base_type = parameter<T>;
    using base_type::backend;
    using base_type::coef0;
    using base_type::cost;
    using base_type::degree;
    using base_type::epsilon;
    using base_type::gamma;
    using base_type::input_filename;
    using base_type::kernel;
    using base_type::model_filename;
    using base_type::predict_filename;
    using base_type::print_info;
    using base_type::target;

    /// The type of the data. Must be either `float` or `double`.
    using real_type = typename base_type::real_type;
    /// Unsigned integer type.
    using size_type = typename base_type::size_type;

    /**
     * @brief Default construct all parameters for prediction.
     */
    parameter_predict() = default;

    /**
     * @brief Set all predict parameters to their default values.
     * @param[in] input_filename the name of the test data file
     * @param[in] model_filename the name of the model file
     */
    explicit parameter_predict(std::string input_filename, std::string model_filename);

    /**
     * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the predict parameters accordingly.
     * @param[in] argc the number of passed command line arguments
     * @param[in] argv the command line arguments
     */
    parameter_predict(int argc, char **argv);
};

extern template class parameter_predict<float>;
extern template class parameter_predict<double>;

}  // namespace plssvm
