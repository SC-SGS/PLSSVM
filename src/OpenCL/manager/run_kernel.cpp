#include "run_kernel.hpp"

namespace opencl {

void run_kernel_1d_timed(device_t &device, cl_kernel kernel, size_t grid_size,
                         size_t block_size) {
  cl_event timing_event = nullptr;
  cl_int err = clEnqueueNDRangeKernel(device.commandQueue, kernel, 1, nullptr,
                                      &grid_size, &block_size, 0, nullptr,
                                      &timing_event);
  check(err, "OCL error: Failed to enqueue kernel command");
  clFinish(device.commandQueue);

  cl_ulong start_time{0};
  cl_ulong end_time{0};
  err = clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_START,
                                sizeof(cl_ulong), &start_time, nullptr);
  check(err, "OCL error: Failed to read start-time from command queue "
             "(or crash in mult)");
  err = clGetEventProfilingInfo(timing_event, CL_PROFILING_COMMAND_END,
                                sizeof(cl_ulong), &end_time, nullptr);
  check(err, "OCL error: Failed to read end-time from command queue");
  clReleaseEvent(timing_event);
  double time = 0.0;
  time = static_cast<double>(end_time - start_time);
  time *= 1e-9;
  std::cout << "duration (s): " << time << std::endl;
}
} // namespace opencl
