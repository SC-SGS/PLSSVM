/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
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
    void run_device_kernel(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, int add);
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace plssvm::openmp