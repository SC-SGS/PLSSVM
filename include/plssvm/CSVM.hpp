#pragma once
#include <plssvm/detail/operators.hpp>
#include <plssvm/kernel_types.hpp>
#include <plssvm/typedef.hpp>

#include "fmt/chrono.h"  // format std::chrono
#include "fmt/core.h"    // fmt::print

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdlib.h>
#include <string>  // std::string
#include <tuple>
#include <type_traits>  // std::is_same_v
#include <utility>
#include <vector>  // std::vector

namespace plssvm {

template <typename T>
class CSVM {
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    using real_type = T;
    using size_type = std::size_t;

    CSVM(real_t cost_, real_t epsilon_, kernel_type kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_ = true) :
        cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), print_info_(info_) {}

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

    const real_t &getB() const { return bias; };
    virtual void load_w() = 0;
    virtual std::vector<real_t> predict(real_t *, int, int) = 0;

  protected:
    // pure virtual, must be implemented by all subclasses
    virtual void loadDataDevice() = 0;
    virtual std::vector<real_t> generate_q() = 0;
    virtual std::vector<real_t> CG(const std::vector<real_t> &b, const int, const real_t, const std::vector<real_t> &q) = 0;

    const bool print_info_;
    real_t cost;
    const real_t epsilon;
    const kernel_type kernel;
    const real_t degree;
    real_t gamma;
    const real_t coef0;
    real_t bias;
    real_t QA_cost;
    std::vector<std::vector<real_t>> data;
    size_t num_features;
    size_t num_data_points;
    std::vector<real_t> value;
    std::vector<real_t> alpha;

    void learn();  // TODO: public after correct exception handling

    real_t kernel_function(std::vector<real_t> &, std::vector<real_t> &);
    real_t kernel_function(real_t *, real_t *, int);

    /**
     * @brief Transforms the 2D data from AoS to a 1D SoA layout, ignoring the last data point and adding boundary points.
     * @param[in] boundary the number of boundary cells
     * @attention boundary values can contain random numbers
     * @return an 1D vector in a SoA layout
     */
    std::vector<real_type> transform_data(const size_type boundary) {
        auto start_time = std::chrono::steady_clock::now();

        std::vector<real_type> vec(num_features * (num_data_points - 1 + boundary));
        #pragma omp parallel for collapse(2)
        for (size_type col = 0; col < num_features; ++col) {
            for (size_type row = 0; row < num_data_points - 1; ++row) {
                vec[col * (num_data_points - 1 + boundary) + row] = data[row][col];
            }
        }

        auto end_time = std::chrono::steady_clock::now();
        if (print_info_) {
            fmt::print("Transformed dataset from 2D AoS to 1D SoA in {}.", std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time));
        }
        return vec;
    }
};

extern template class CSVM<float>;
extern template class CSVM<double>;

}  // namespace plssvm