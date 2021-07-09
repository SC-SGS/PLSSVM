#pragma once

#include "plssvm/CSVM.hpp"
#include "plssvm/backends/OpenCL/DevicePtrOpenCL.hpp"
#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type
#include "plssvm/parameter.hpp"     // plssvm::parameter

#include "../../../../src/plssvm/backends/OpenCL/manager/configuration.hpp"
#include "../../../../src/plssvm/backends/OpenCL/manager/device.hpp"
#include "../../../../src/plssvm/backends/OpenCL/manager/manager.hpp"

#include <vector>  // std::vector

namespace plssvm {

template <typename T>
class OpenCL_CSVM : public CSVM<T> {
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

    explicit OpenCL_CSVM(parameter<T> &params);
    OpenCL_CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

    //    void std::vector<real_type> predict(real_type *, size_type, size_type) override;  // TODO: implement

  protected:
    void setup_data_on_device() override;
    std::vector<real_type> generate_q() override;
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) override;
    //    void load_w() override;  // TODO: implement

    opencl::manager_t manager{ "../platform_configuration.cfg" };
    opencl::device_t first_device;
    std::vector<cl_kernel> kernel_q_cl;
    std::vector<cl_kernel> svm_kernel_linear;
    std::vector<opencl::DevicePtrOpenCL<real_type>> datlast_cl;
    std::vector<opencl::DevicePtrOpenCL<real_type>> data_cl;
};

extern template class OpenCL_CSVM<float>;
extern template class OpenCL_CSVM<double>;

}  // namespace plssvm