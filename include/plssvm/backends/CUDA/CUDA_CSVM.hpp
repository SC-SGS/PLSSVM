#pragma once

#include "plssvm/CSVM.hpp"
#include "plssvm/backends/CUDA/CUDA_DevicePtr.cuh"  // plssvm::detail::cuda::device_ptr
#include "plssvm/kernel_types.hpp"                  // plssvm::kernel_type
#include "plssvm/parameter.hpp"                     // plssvm::parameter

namespace plssvm {

template <typename T>
class CUDA_CSVM : public CSVM<T> {
  protected:
    // protected for test MOCK class
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
    using real_type = typename base_type::real_type;
    using size_type = typename base_type::size_type;

    explicit CUDA_CSVM(const parameter<T> &params);
    CUDA_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

    std::vector<real_type> predict(const real_type *, size_type, size_type);  // TODO: implement correctly, add override

  protected:
    void setup_data_on_device() override;
    std::vector<real_type> generate_q() override;
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) override;
    void load_w() override;  // TODO: implement correctly

    void run_device_kernel(int device, const detail::cuda::device_ptr<real_type> &q_d, detail::cuda::device_ptr<real_type> &r_d, const detail::cuda::device_ptr<real_type> &x_d, const detail::cuda::device_ptr<real_type> &data_d, int sign);

    const int num_devices_{};
    std::vector<detail::cuda::device_ptr<real_type>> data_d_{};
    std::vector<detail::cuda::device_ptr<real_type>> data_last_d_{};
    detail::cuda::device_ptr<real_type> w_d_{};
};

extern template class CUDA_CSVM<float>;
extern template class CUDA_CSVM<double>;

}  // namespace plssvm
