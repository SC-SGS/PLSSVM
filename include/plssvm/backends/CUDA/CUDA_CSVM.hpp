#pragma once

#include "plssvm/CSVM.hpp"
#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type
#include "plssvm/parameter.hpp"     // plssvm::parameter

namespace plssvm {

template <typename T>
class CUDA_CSVM : public CSVM<T> {
  protected:
    // protected for test MOCK class
    using base_type = CSVM<T>;
    using base_type::alpha_;
    using base_type::cost_;
    using base_type::data_;
    using base_type::kernel_;
    using base_type::num_data_points_;
    using base_type::num_features_;
    using base_type::print_info_;
    using base_type::QA_cost_;

  public:
    using real_type = typename base_type::real_type;
    using size_type = typename base_type::size_type;

    explicit CUDA_CSVM(parameter<T> &params);
    CUDA_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

    void std::vector<real_type> predict(const real_type *, size_type, size_type) override;  // TODO: implement correctly

  protected:
    void setup_data_on_device() override;
    std::vector<real_type> generate_q() override;
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) override;
    void load_w() override;  // TODO: implement correctly

    std::vector<real_type *> data_d;
    std::vector<real_type *> datlast_d;
    real_type *w_d;
};

extern template class CUDA_CSVM<float>;
extern template class CUDA_CSVM<double>;

}  // namespace plssvm
