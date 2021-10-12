/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a C-SVM using the OpenMP backend.
 */

#pragma once

#include "plssvm/csvm.hpp"             // plssvm::csvm
#include "plssvm/kernel_types.hpp"     // plssvm::kernel_type
#include "plssvm/parameter.hpp"        // plssvm::parameter
#include "plssvm/target_platform.hpp"  // plssvm::target_platform

#include <vector>  // std::vector

namespace plssvm::openmp {

/**
 * @brief The C-SVM class using the OpenMP backend.
 * @tparam T the type of the data
 */
template <typename T>
class csvm : public ::plssvm::csvm<T> {
  protected:
    // protected for test mock class
    /// The template base type of the OpenMP C-SVM class.
    using base_type = ::plssvm::csvm<T>;
    using base_type::alpha_ptr_;
    using base_type::bias_;
    using base_type::coef0_;
    using base_type::cost_;
    using base_type::data_ptr_;
    using base_type::degree_;
    using base_type::gamma_;
    using base_type::kernel_;
    using base_type::num_data_points_;
    using base_type::num_features_;
    using base_type::print_info_;
    using base_type::QA_cost_;
    using base_type::target_;
    using base_type::w_;

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = typename base_type::real_type;
    /// Unsigned integer type.
    using size_type = typename base_type::size_type;

    /**
     * @brief Construct a new C-SVM using the OpenMP backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit csvm(const parameter<T> &params);

    /**
     * @brief Uses the already learned model to predict the class of multiple (new) data points.
     * @param[in] points the data points to predict
     * @return a `std::vector<real_type>` filled with negative values for each prediction for a data point with the negative class and positive values otherwise ([[nodiscard]])
     */
    [[nodiscard]] virtual std::vector<real_type> predict(const std::vector<std::vector<real_type>> &points) override;

  protected:
    void setup_data_on_device() override {
        // OpenMP device is the CPU -> no special load functions
    }
    std::vector<real_type> generate_q() override;
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) override;

    /**
     * @brief Select the correct kernel based on the value of @p kernel_ and run it on the CPU using OpenMP.
     * @param[in] q the `q` vector
     * @param[out] ret the result vector
     * @param[in] d the right-hand side
     * @param[in] data the data
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    void run_device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type add);

    /**
     * @brief updates the `w_` vector to the current data and alpha values.
     */
    virtual void update_w() override;
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace plssvm::openmp