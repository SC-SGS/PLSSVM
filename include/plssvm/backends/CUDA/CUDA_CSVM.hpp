/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines a C-SVM using the CUDA backend.
 */

#pragma once

#include "plssvm/CSVM.hpp"                             // plssvm::CSVM
#include "plssvm/backends/CUDA/detail/device_ptr.cuh"  // plssvm::cuda::detail::device_ptr
#include "plssvm/kernel_types.hpp"                     // plssvm::kernel_type
#include "plssvm/parameter.hpp"                        // plssvm::parameter

#include <vector>  // std::vector

namespace plssvm {

/**
 * @brief The C-SVM class using the CUDA backend.
 * @tparam T the type of the data
 */
template <typename T>
class CUDA_CSVM : public CSVM<T> {
  protected:
    // protected for the test MOCK class
    /// The template base type of the CUDA_SVM class.
    using base_type = CSVM<T>;
    using base_type::alpha_;
    using base_type::coef0_;
    using base_type::cost_;
    using base_type::data_;
    using base_type::degree_;
    using base_type::gamma_;
    using base_type::kernel_;
    using base_type::num_data_points_;
    using base_type::num_features_;
    using base_type::print_info_;
    using base_type::QA_cost_;

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = typename base_type::real_type;
    /// Unsigned integer type.
    using size_type = typename base_type::size_type;

    /**
     * @brief Construct a new C-SVM using the CUDA backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit CUDA_CSVM(const parameter<T> &params);
    /**
     * @brief Construct an new C-SVM using the CUDA backend explicitly specifying all necessary parameters.
     * @param[in] kernel the type of the kernel function
     * @param[in] degree parameter used in the polynomial kernel function
     * @param[in] gamma parameter used in the polynomial and rbf kernel functions
     * @param[in] coef0 parameter use din the polynomial kernel function
     * @param[in] cost parameter of the C-SVM
     * @param[in] epsilon error tolerance in the CG algorithm
     * @param[in] print_info if `true` additional information will be printed during execution
     */
    CUDA_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

    /**
     * @brief TODO: predict
     * @return
     */
    std::vector<real_type> predict(const real_type *, size_type, size_type);  // TODO: implement correctly, add override

  protected:
    void setup_data_on_device() override;
    std::vector<real_type> generate_q() override;
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) override;
    void load_w() override;  // TODO: implement correctly

    /**
     * @brief Select the correct kernel based on the value of @p kernel_ and run it on the CUDA @p device.
     * @param[in] device the CUDA device to run the kernel on
     * @param[in] q_d subvector of the least-squares matrix equation
     * @param[inout] r_d the result vector
     * @param[in] x_d the `x` vector
     * @param[in] data_d the data
     * @param[in] add denotes whether the values are added or subtracted from the result vector
     */
    void run_device_kernel(int device, const cuda::detail::device_ptr<real_type> &q_d, cuda::detail::device_ptr<real_type> &r_d, const cuda::detail::device_ptr<real_type> &x_d, const cuda::detail::device_ptr<real_type> &data_d, int add);
    /**
     * @brief Combines the data in @p buffer_d from all devices into @p buffer and distributes them back to each devices.
     * @param[inout] buffer_d the data to gather
     * @param[inout] buffer the reduces data
     */
    void device_reduction(std::vector<cuda::detail::device_ptr<real_type>> &buffer_d, std::vector<real_type> &buffer);

    /// The number of available/used CUDA devices.
    int num_devices_{};
    /// The data saved across all devices.
    std::vector<cuda::detail::device_ptr<real_type>> data_d_{};
    /// The last row of the data matrix.
    std::vector<cuda::detail::device_ptr<real_type>> data_last_d_{};
    /// TODO:
    cuda::detail::device_ptr<real_type> w_d_{};
};

extern template class CUDA_CSVM<float>;
extern template class CUDA_CSVM<double>;

}  // namespace plssvm
