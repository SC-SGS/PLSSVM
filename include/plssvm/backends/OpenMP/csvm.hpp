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
#include "plssvm/parameter.hpp"  // plssvm::parameter
#include "plssvm/target_platforms.hpp"

#include <vector>  // std::vector

namespace plssvm::openmp {

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

  public:
    /// The type of the data. Must be either `float` or `double`.
    using typename base_type::real_type;
    using typename base_type::size_type;

    /**
     * @brief Construct a new C-SVM using the OpenMP backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     * @throws plssvm::csvm::csvm() exceptions
     * @throws plssvm::openmp::backend_exception if the target platform isn't plssvm::target_platform::automatic or plssvm::target_platform::cpu
     * @throws plssvm::openmp::backend_exception if the plssvm::target_platform::cpu target isn't available
     */
    explicit csvm(target_platform target, parameter<T> params = {});
    template <typename... Args>
    csvm(target_platform target, kernel_type kernel, Args&&... named_args) : base_type{ kernel, std::forward<Args>(named_args)... } {
        this->init(target);
    }

  private:
    /**
     * @copydoc plssvm::csvm::setup_data_on_device
     */
    void setup_data_on_device() override {
        // OpenMP device is the CPU -> no special load functions
    }
    /**
     * @copydoc plssvm::csvm::clear_data_from_device
     */
    void clear_data_from_device() override {
        // OpenMP device is the CPU -> no special clear functions
    }
    /**
     * @copydoc plssvm::csvm::generate_q
     */
    [[nodiscard]] std::vector<real_type> generate_q(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &data) override;
    /**
     * @copydoc plssvm::csvm::solver_CG
     */
    [[nodiscard]] std::vector<real_type> conjugate_gradient(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &A, const std::vector<real_type> &b, const std::vector<real_type> &q, real_type QA_cost, real_type eps, size_type max_iter) override;
    /**
     * @copydoc plssvm::csvm::calculate_w
     */
    [[nodiscard]] std::vector<real_type> calculate_w(const std::vector<std::vector<real_type>> &A, const std::vector<real_type> &alpha) override;
    /**
     * @copydoc plssvm::csvm::predict_values_impl
     */
    [[nodiscard]] std::vector<real_type> predict_values_impl(const parameter<real_type> &params, const std::vector<std::vector<real_type>> &support_vectors, const std::vector<real_type> &alpha, real_type rho, std::shared_ptr<std::vector<real_type>> &w, const std::vector<std::vector<real_type>> &predict_points) override;


    /**
     * @brief Select the correct kernel based on the value of @p kernel_ and run it on the CPU using OpenMP.
     * @param[in] q the `q` vector
     * @param[out] ret the result vector
     * @param[in] d the right-hand side of the equation
     * @param[in] data the data
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    void run_device_kernel(const parameter<real_type> &params, const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type QA_cost, real_type add);

  private:
    /**
     * @brief Initializes the OpenMP backend and performs some sanity checks.
     * @param[in] target the platform to run on (must be `plssvm::target_platfrom::cpu` for the OpenMP backend).
     */
    void init(target_platform target);
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace plssvm::openmp