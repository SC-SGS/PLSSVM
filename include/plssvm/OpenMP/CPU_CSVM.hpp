#pragma once

#include <plssvm/CSVM.hpp>

class CPU_CSVM : public CSVM {
  public:
    CPU_CSVM(real_t cost_, real_t epsilon_, unsigned kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_);

    void learn(std::string &, std::string &);

    void load_w(){};                                   // TODO: implement load_w
    std::vector<real_t> predict(real_t *, int, int){}; //TODO: implement predict

  private:
    std::vector<real_t> CG(const std::vector<real_t> &b, const int, const real_t);
    void loadDataDevice(){};
    void learn();
};
