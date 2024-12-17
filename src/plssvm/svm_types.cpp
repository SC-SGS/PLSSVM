/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/svm_types.hpp"

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::to_lower_case
#include "plssvm/detail/utility.hpp"         // plssvm::detail::contains
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::unsupported_backend_exception
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

#include "fmt/format.h"  // fmt::format
#include "fmt/ranges.h"  // fmt::join

#include <array>        // std::array
#include <ios>          // std::ios::failbit
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::pair
#include <vector>       // std::vector

namespace plssvm {

std::vector<svm_type> list_available_svm_types() {
    return { svm_type::csvc, svm_type::csvr };
}

std::ostream &operator<<(std::ostream &out, const svm_type svm) {
    switch (svm) {
        case svm_type::csvc:
            return out << "csvc";
        case svm_type::csvr:
            return out << "csvr";
    }
    return out << "unknown";
}

std::string_view svm_type_to_task_name(const svm_type svm) noexcept {
    switch (svm) {
        case svm_type::csvc:
            return "classification";
        case svm_type::csvr:
            return "regression";
    }
    return "unknown";
}

std::istream &operator>>(std::istream &in, svm_type &svm) {
    std::string str;
    in >> str;
    detail::to_lower_case(str);

    if (str == "csvc" || str == "c-svc" || str == "c_svc" || str == "0") {
        svm = svm_type::csvc;
    } else if (str == "csvr" || str == "c-svr" || str == "c_svr" || str == "1") {
        svm = svm_type::csvr;
    } else {
        in.setstate(std::ios::failbit);
    }
    return in;
}

}  // namespace plssvm
