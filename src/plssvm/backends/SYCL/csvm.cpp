/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 */

#include "plssvm/backends/SYCL/csvm.hpp"

#include "plssvm/csvm.hpp"          // plssvm::csvm
#include "plssvm/kernel_types.hpp"  // plssvm::kernel_types
#include "plssvm/parameter.hpp"     // plssvm::parameter

#include "CL/sycl.hpp"  // SYCL stuff

#include "fmt/core.h"  // fmt::print

#include <vector>  // std::vector

namespace plssvm::sycl {

template <typename T>
csvm<T>::csvm(const parameter<T> &params) :
    csvm{ params.kernel, params.degree, params.gamma, params.coef0, params.cost, params.epsilon, params.print_info } {}

template <typename T>
csvm<T>::csvm(const kernel_type kernel, const real_type degree, const real_type gamma, const real_type coef0, const real_type cost, const real_type epsilon, const bool print_info) :
    ::plssvm::csvm<T>{ kernel, degree, gamma, coef0, cost, epsilon, print_info } {
    if (print_info_) {
        fmt::print("Using SYCL as backend.\n");
    }

    ::sycl::queue q{ ::sycl::gpu_selector{} };
    fmt::print("{}\n", q.get_device().get_info<::sycl::info::device::name>());
}

template <typename T>
void csvm<T>::setup_data_on_device() {
}

template <typename T>
auto csvm<T>::generate_q() -> std::vector<real_type> {
    return {};
}

template <typename T>
auto csvm<T>::solver_CG(const std::vector<real_type> &b, const size_type imax, const real_type eps, const std::vector<real_type> &q) -> std::vector<real_type> {
    return {};
}

template class csvm<float>;
template class csvm<double>;

}  // namespace plssvm::sycl