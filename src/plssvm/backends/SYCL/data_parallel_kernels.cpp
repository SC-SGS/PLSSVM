/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/data_parallel_kernels.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case

#include <ios>      // std::ios::failbit
#include <istream>  // std::istream
#include <ostream>  // std::ostream
#include <string>   // std::string
#include <vector>   // std::vector

namespace plssvm::sycl {

std::vector<data_parallel_kernel> list_available_sycl_data_parallel_kernels() {
    std::vector<data_parallel_kernel> available_sycl_data_parallel_kernels = {
        data_parallel_kernel::automatic,
        data_parallel_kernel::basic,
        data_parallel_kernel::work_group
    };
#if defined(PLSSVM_SYCL_HIERARCHICAL_AND_SCOPED_KERNELS_ENABLED)
    available_sycl_data_parallel_kernels.push_back(data_parallel_kernel::hierarchical);
    #if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    available_sycl_data_parallel_kernels.push_back(data_parallel_kernel::scoped);
    #endif
#endif
    return available_sycl_data_parallel_kernels;
}

std::ostream &operator<<(std::ostream &out, const data_parallel_kernel kernel_type) {
    switch (kernel_type) {
        case data_parallel_kernel::automatic:
            return out << "automatic";
        case data_parallel_kernel::basic:
            return out << "basic";
        case data_parallel_kernel::work_group:
            return out << "work_group";
        case data_parallel_kernel::hierarchical:
            return out << "hierarchical";
        case data_parallel_kernel::scoped:
            return out << "scoped";
    }
    return out << "unknown";
}

std::istream &operator>>(std::istream &in, data_parallel_kernel &kernel_type) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "automatic" || str == "auto") {
        kernel_type = data_parallel_kernel::automatic;
    } else if (str == "basic") {
        kernel_type = data_parallel_kernel::basic;
    } else if (str == "work_group" || str == "work-group" || str == "nd_range" || str == "nd-range") {
        kernel_type = data_parallel_kernel::work_group;
    } else if (str == "hierarchical") {
        kernel_type = data_parallel_kernel::hierarchical;
    } else if (str == "scoped") {
        kernel_type = data_parallel_kernel::scoped;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm::sycl
