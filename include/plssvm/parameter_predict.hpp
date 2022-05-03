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
    using base_type::kernel;
    using base_type::sycl_implementation_type;
    using base_type::target;

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

    /// The name of the data/test file to parse.
    std::string input_filename{};
    /// The name of the model file to write the learned support vectors to/to parse the saved model from.
    std::string model_filename{};
    /// The name of the file to write the prediction to.
    std::string predict_filename{};

  private:
    /**
     * @brief Generate a predict filename based on the name of the input file.
     * @return `${input_filename}.predict` (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string predict_name_from_input();
};

extern template class parameter_predict<float>;
extern template class parameter_predict<double>;

template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter_predict<T> &params);

}  // namespace plssvm
