#pragma once

#include "manager.hpp"
#include <CL/cl.h>

// #include "print_type.hpp"

namespace opencl {
namespace detail {
template <int32_t arg_num> void apply_arguments(cl_kernel kernel) {}

template <int32_t arg_num, typename T, typename... Ts>
void apply_arguments(cl_kernel kernel, T first_arg, Ts... remaining_args) {
  // std::cout << "type name for arg num: " << arg_num
  //           << " name: " << type_name<T>() << std::endl;
  cl_int err = clSetKernelArg(kernel, arg_num, sizeof(T), &first_arg);
  check(err,
        "error applying argument " + std::to_string(arg_num) + " to kernel");
  detail::apply_arguments<arg_num + 1>(kernel, remaining_args...);
}
} // namespace detail

template <typename... Ts> void apply_arguments(cl_kernel kernel, Ts... args) {
  detail::apply_arguments<0>(kernel, args...);
}

} // namespace opencl
