#pragma once

#include "plssvm/parameter.hpp"  // plssvm::parameter
#include <plssvm/CSVM.hpp>
#include <plssvm/kernel_types.hpp>
#include <plssvm/typedef.hpp>

namespace plssvm {

class CUDA_CSVM : public CSVM<real_t> {
  public:
    explicit CUDA_CSVM(parameter<real_t> &params);
    CUDA_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

    void load_w();
    std::vector<real_t> predict(real_t *, int, int);

  protected:
    void setup_data_on_device() override;
    std::vector<real_t> generate_q() override;
    std::vector<real_t> solver_CG(const std::vector<real_t> &b, std::size_t, real_t, const std::vector<real_t> &q) override;
    std::vector<real_t *> data_d;
    std::vector<real_t *> datlast_d;
    real_t *w_d;
};

}  // namespace plssvm
