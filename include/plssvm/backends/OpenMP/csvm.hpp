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

#include "plssvm/csvm.hpp"  // plssvm::csvm

#include <vector>  // std::vector

namespace plssvm {

// forward declare parameter class
template <typename T>
class parameter;

namespace openmp {

/**
 * @brief A C-SVM implementation using OpenMP as backend.
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
    /**
     * @copydoc plssvm::csvm::predict(const std::vector<real_type>&)
     */
    using base_type::predict;

    using typename base_type::real_type;

    /**
     * @brief Construct a new C-SVM using the OpenMP backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::csvm::csvm() exceptions
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    explicit csvm(const parameter<T> &params);

    /**
     * @copydoc plssvm::csvm::predict(const std::vector<std::vector<real_type>>&)
     */
    [[nodiscard]] std::vector<real_type> predict(const std::vector<std::vector<real_type>> &points) override;

  protected:
    /**
     * @copydoc plssvm::csvm::setup_data_on_device
     */
    void setup_data_on_device() override {
        // OpenMP device is the CPU -> no special load functions
    }
    /**
     * @copydoc plssvm::csvm::generate_q
     */
    [[nodiscard]] std::vector<real_type> generate_q() override;
    /**
     * @copydoc plssvm::csvm::solver_CG
     */
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, std::size_t imax, real_type eps, const std::vector<real_type> &q) override;
    /**
     * @copydoc plssvm::csvm::update_w
     */
    void update_w() override;

    /**
     * @brief Select the correct kernel based on the value of @p kernel_ and run it on the CPU using OpenMP.
     * @param[in] q the `q` vector
     * @param[out] ret the result vector
     * @param[in] d the right-hand side of the equation
     * @param[in] data the data
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    void run_device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type add);
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace openmp
}  // namespace plssvm