#pragma once

#include <plssvm/CSVM.hpp>
#include <plssvm/kernel_types.hpp>

namespace plssvm {

class OpenMP_CSVM : public CSVM {
  public:
    OpenMP_CSVM(real_t cost_, real_t epsilon_, kernel_type kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_);

    virtual void learn(std::string &, std::string &);

    virtual void load_w(){};                                   // TODO: implement load_w
    virtual std::vector<real_t> predict(real_t *, int, int){}; //TODO: implement predict
    virtual void learn();

  protected:
    virtual std::vector<real_t> CG(const std::vector<real_t> &b, const int, const real_t);
    void loadDataDevice(){};
};

} // namespace plssvm