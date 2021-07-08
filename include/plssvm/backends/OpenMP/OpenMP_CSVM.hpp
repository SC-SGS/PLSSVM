#pragma once

#include <plssvm/CSVM.hpp>
#include <plssvm/kernel_types.hpp>
#include <plssvm/typedef.hpp>

namespace plssvm {

class OpenMP_CSVM : public CSVM<real_t> {
  public:
    OpenMP_CSVM(real_t cost_, real_t epsilon_, kernel_type kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_);

    virtual void load_w(){};                                                 // TODO: implement load_w
    virtual std::vector<real_t> predict(real_t *, int, int) { return {}; };  //TODO: implement predict

  protected:
    void setup_data_on_device() override {
        // OpenMP device is the CPU -> no special load functions
    }
    std::vector<real_t> generate_q() override;
    std::vector<real_t> solver_CG(const std::vector<real_t> &b, std::size_t, real_t, const std::vector<real_t> &q) override;
};

}  // namespace plssvm