#pragma once
#include <plssvm/operators.hpp>
#include <plssvm/typedef.hpp>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <math.h>
#include <omp.h>
#include <sstream>
#include <stdlib.h>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace plssvm {

const bool times = 0;

// static const unsigned CUDABLOCK_SIZE = 7;
// static const unsigned BLOCKING_SIZE_THREAD = 2;

class CSVM {
  public:
    CSVM(real_t cost_, real_t epsilon_, unsigned kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_) : cost(cost_), epsilon(epsilon_), kernel(kernel_), degree(degree_), gamma(gamma_), coef0(coef0_), info(info_) {}
    virtual void learn(const std::string_view, const std::string_view);

    const real_t &getB() const { return bias; };
    virtual void load_w() = 0;
    virtual std::vector<real_t> predict(real_t *, int, int) = 0;
    virtual ~CSVM() = default;

    void libsvmParser(const std::string_view);
    void arffParser(const std::string_view);
    void writeModel(const std::string_view);

  protected:
    const bool info;
    real_t cost;
    const real_t epsilon;
    const unsigned kernel;
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

    virtual void learn();

    real_t kernel_function(std::vector<real_t> &, std::vector<real_t> &);
    real_t kernel_function(real_t *, real_t *, int);

    virtual void loadDataDevice() = 0;

    virtual std::vector<real_t> CG(const std::vector<real_t> &b, const int, const real_t) = 0;

    /**
     * @brief Transforms the data matrix in SoA, while it ignores the last datapoint and adds boundary places,
     * @attention boundary values can contain random numbers
     * @param boundary the number of boundary cells
     * @return std::vector<real_t> SoA
     */
    std::vector<real_t> transform_data(const int boundary) {
        std::vector<real_t> vec(num_features * (num_data_points - 1 + boundary));
#pragma omp parallel for collapse(2)
        for (size_t col = 0; col < num_features; ++col) {
            for (size_t row = 0; row < num_data_points - 1; ++row) {
                vec[col * (num_data_points - 1 + boundary) + row] = data[row][col];
            }
        }
        return vec;
    }
    void loadDataDevice(const int device, const int boundary, const int start_line, const int number_lines, const std::vector<real_t> data);
};

} // namespace plssvm