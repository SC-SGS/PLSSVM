#pragma once

#include <CL/cl.h>
#include <string>

namespace opencl {
void report_error(cl_int err);
std::string resolve_error(cl_int err);
} // namespace opencl
