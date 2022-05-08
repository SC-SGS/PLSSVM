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

#include "plssvm/parameter.hpp"  // plssvm::parameter

#include <string>  // std::string

namespace plssvm {

/**
 * @brief Class for encapsulating all necessary parameters for scaling a data set possibly provided through command line arguments.
 * @tparam T the type of the data
 */
template <typename T>
class parameter_scale : public parameter<T> {
  public:
    /// The template base type of the parameter_train class.
    using base_type = parameter<T>;

    using typename base_type::real_type;

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

    // TODO: mote LIBSVM conform?
    real_type lower{ -1 };
    real_type upper{ +1 };

    /// The name of the data/test file to parse.
    std::string input_filename{};
    /// The name of the model file to write the learned support vectors to/to parse the saved model from.
    std::string scaled_filename{};

  private:
    /**
     * @brief Generate a scaled filename based on the name of the input file.
     * @return `${input_filename}.scaled` (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string scaled_name_from_input();
};

extern template class parameter_scale<float>;
extern template class parameter_scale<double>;

template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter_scale<T> &params);

}  // namespace plssvm
