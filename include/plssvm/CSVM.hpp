#pragma once

#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type
#include "plssvm/parameter.hpp"     // plssvm::parameter

#include <cstddef>      // std::size_t
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

    explicit CSVM(parameter<T> &params);
    CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

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
    virtual std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) = 0;
    //    virtual void load_w() = 0; // TODO: implemented together with predict

    /**
     * @brief Transforms the 2D data from AoS to a 1D SoA layout, ignoring the last data point and adding boundary points.
     * @param[in] boundary the number of boundary cells
     * @attention boundary values can contain random numbers
     * @return an 1D vector in a SoA layout
     */
    std::vector<real_type> transform_data(size_type boundary);

    // kernel functions: linear, polynomial, rbf
    real_type kernel_function(const real_type *xi, const real_type *xj, size_type dim);
    real_type kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj);

    // parameter initialized by the constructor
    const kernel_type kernel_;
    const real_type degree_;
    real_type gamma_;
    const real_type coef0_;
    real_type cost_;
    const real_type epsilon_;
    const bool print_info_ = true;

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