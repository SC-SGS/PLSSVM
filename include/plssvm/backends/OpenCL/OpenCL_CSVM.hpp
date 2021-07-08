#pragma once

#include <plssvm/CSVM.hpp>
#include <plssvm/kernel_types.hpp>
#include <plssvm/typedef.hpp>

#include "../../../../src/plssvm/backends/OpenCL/manager/configuration.hpp"
#include "../../../../src/plssvm/backends/OpenCL/manager/device.hpp"
#include "../../../../src/plssvm/backends/OpenCL/manager/manager.hpp"
#include <plssvm/backends/OpenCL/DevicePtrOpenCL.hpp>
#include <stdexcept>

namespace plssvm {

class OpenCL_CSVM : public CSVM<real_t> {
  public:
    OpenCL_CSVM(real_t cost_, real_t epsilon_, kernel_type kernel_, real_t degree_, real_t gamma_, real_t coef0_, bool info_);
    void load_w(){};                                                 // TODO: implement load_w
    std::vector<real_t> predict(real_t *, int, int) { return {}; };  //TODO: implement predict

  protected:
    void setup_data_on_device() override;
    std::vector<real_t> CG(const std::vector<real_t> &b, const int, const real_t, const std::vector<real_t> &);
    void resizeData(int boundary);
    void resizeData(const int device, int boundary);
    void resizeDatalast(int boundary);
    void resizeDatalast(const int device, int boundary);
    std::vector<real_t> generate_q();
    // inline void resize(const int old_boundary,const int new_boundary);
    opencl::manager_t manager{ "../platform_configuration.cfg" };
    opencl::device_t first_device;
    std::vector<cl_kernel> kernel_q_cl;
    std::vector<cl_kernel> svm_kernel_linear;
    std::vector<opencl::DevicePtrOpenCL<real_t>> datlast_cl;
    std::vector<opencl::DevicePtrOpenCL<real_t>> data_cl;
};

}  // namespace plssvm