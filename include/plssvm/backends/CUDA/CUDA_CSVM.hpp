#pragma once

#include <plssvm/CSVM.hpp>
#include <plssvm/kernel_types.hpp>

namespace plssvm {

class CUDA_CSVM : public CSVM {
  public:
    CUDA_CSVM(real_t cost_, real_t epsilon_, kernel_type kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_);
    void loadDataDevice();
    void load_w();
    std::vector<real_t> predict(real_t *, int, int);

  protected:
    std::vector<real_t> CG(const std::vector<real_t> &b, const int, const real_t);

    std::vector<real_t *> data_d;
    std::vector<real_t *> datlast_d;
    real_t *w_d;
};

}  // namespace plssvm
