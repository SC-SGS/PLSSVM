#pragma once

#include <plssvm/kernel_types.hpp>  // plssvm::kernel_type

#include "fmt/chrono.h"  // format std::chrono
#include "fmt/core.h"    // fmt::print

#include <chrono>       // std::chrono::stead_clock, std::chrono::duration_cast, std::chrono::milliseconds
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

namespace plssvm {

template <typename T = double>
class CSVM {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    using real_type = T;
    using size_type = std::size_t;

    CSVM(real_type cost, real_type epsilon, kernel_type kernel, real_type degree, real_type gamma, real_type coef0, bool info = true) :
        cost_(cost), epsilon_(epsilon), kernel_(kernel), degree_(degree), gamma_(gamma), coef0_(coef0), print_info_(info) {}

    virtual ~CSVM() = default;

    /**************************************************************************************************************************************/
    /**                                                          IO functions                                                            **/
    /**************************************************************************************************************************************/
    void parse_libsvm(const std::string &filename);
    void parse_arff(const std::string &filename);
    void parse_file(const std::string &filename);
    void write_model(const std::string &filename);

    /**************************************************************************************************************************************/
    /**                                                              learn                                                               **/
    /**************************************************************************************************************************************/
    void learn(const std::string &input_filename, const std::string &model_filename);

  protected:
    void learn();  // TODO: public after correct exception handling

  public:
    /**************************************************************************************************************************************/
    /**                                                             predict                                                              **/
    /**************************************************************************************************************************************/
    // TODO: protected?
    //    virtual std::vector<real_type> predict(real_type *, size_type, size_type) = 0;

    /**************************************************************************************************************************************/
    /**                                                              getter                                                              **/
    /**************************************************************************************************************************************/
    // TODO: other getter?
    //    [[nodiscard]] real_type get_bias() const noexcept { return bias_; };

  protected:
    // pure virtual, must be implemented by all subclasses
    virtual void setup_data_on_device() = 0;
    virtual std::vector<real_type> generate_q() = 0;
    virtual std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type, real_type, const std::vector<real_type> &q) = 0;
    //    virtual void load_w() = 0; // TODO: implemented together with predict

    // kernel functions: linear, polynomial, rbf // TODO: move somewhere else?
    real_type kernel_function(const std::vector<real_type> &, const std::vector<real_type> &);
    real_type kernel_function(const real_type *, const real_type *, size_type);

    /**
 * @brief Transforms the 2D data from AoS to a 1D SoA layout, ignoring the last data point and adding boundary points.
 * @param[in] boundary the number of boundary cells
 * @attention boundary values can contain random numbers
 * @return an 1D vector in a SoA layout
 */
    std::vector<real_type> transform_data(const size_type boundary) {
        auto start_time = std::chrono::steady_clock::now();

        std::vector<real_type> vec(num_features_ * (num_data_points_ - 1 + boundary));
        #pragma omp parallel for collapse(2)
        for (size_type col = 0; col < num_features_; ++col) {
            for (size_type row = 0; row < num_data_points_ - 1; ++row) {
                vec[col * (num_data_points_ - 1 + boundary) + row] = data_[row][col];
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        if (print_info_) {
            fmt::print("Transformed dataset from 2D AoS to 1D SoA in {}.", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
        }
        return vec;
    }

    // parameter initialized by the constructor
    real_type cost_;
    const real_type epsilon_;
    const kernel_type kernel_;
    const real_type degree_;
    real_type gamma_;
    const real_type coef0_;
    const bool print_info_;

    // internal variables
    size_type num_data_points_{};
    size_type num_features_{};
    std::vector<std::vector<real_type>> data_{};
    std::vector<real_type> value_{};
    real_type bias_{};
    real_type QA_cost_{};
    std::vector<real_type> alpha_{};
};

extern template class CSVM<float>;
extern template class CSVM<double>;

}  // namespace plssvm