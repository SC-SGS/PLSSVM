#pragma once

#include "plssvm/CSVM.hpp"
#include "plssvm/kernel_types.hpp"
#include "plssvm/parameter.hpp"  // plssvm::parameter

#include <vector>

namespace plssvm {

template <typename T>
class OpenMP_CSVM : public CSVM<T> {
  protected:
    // protected for test MOCK class
    using base_type = CSVM<T>;
    using base_type::cost_;
    using base_type::data_;
    using base_type::print_info_;
    using base_type::QA_cost_;

  public:
    using real_type = typename base_type::real_type;
    using size_type = typename base_type::size_type;

    explicit OpenMP_CSVM(parameter<T> &params);
    OpenMP_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

    //    void std::vector<real_type> predict(real_type *, size_type, size_type) override;  // TODO: implement

  protected:
    void setup_data_on_device() override {
        // OpenMP device is the CPU -> no special load functions
    }
    std::vector<real_type> generate_q() override;
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type, real_type, const std::vector<real_type> &q) override;
    //    void load_w() override;  // TODO: implement
};

extern template class OpenMP_CSVM<float>;
extern template class OpenMP_CSVM<double>;

}  // namespace plssvm