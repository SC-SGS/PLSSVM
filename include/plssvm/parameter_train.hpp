/**
* @file
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright
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
    using base_type::input_filename;
    using base_type::kernel;
    using base_type::model_filename;
    using base_type::predict_filename;
    using base_type::print_info;
    using base_type::target;

    using real_type = typename base_type::real_type;
    using size_type = typename base_type::size_type;

    /**
     * @brief Set all training parameters to their default values.
     * @param[in] input_filename the name of the data file
     */
    explicit parameter_train(std::string input_filename);

    /**
     * @brief Parse the command line arguments @p argv using [`cxxopts`](https://github.com/jarro2783/cxxopts) and set the training parameters accordingly.
     * @param[in] argc the number of passed command line arguments
     * @param[in] argv the command line arguments
     */
    parameter_train(int argc, char **argv);

  private:
    /*
     * Generate model filename based on the name of the input file.
     */
    void model_name_from_input();
};

extern template class parameter_train<float>;
extern template class parameter_train<double>;

}  // namespace plssvm
