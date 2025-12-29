/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/svm_types.hpp"

#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::{starts_with, to_lower_case}
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_file_format_exception

#include "fmt/format.h"  // fmt::format

#include <fstream>      // std::ifstream
#include <ios>          // std::ios::failbit
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <sstream>      // std::istringstream
#include <string>       // std::string, std::getline
#include <string_view>  // std::string_view
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

svm_type svm_type_from_model_file(const std::string &filename) {
    // open the model file and check for the used SVM type
    std::ifstream model_file{ filename };
    // check if the file could be opened successfully
    if (model_file.fail()) {
        throw invalid_file_format_exception{ fmt::format("The provided model file \"{}\" can't be opened!", filename) };
    }

    std::string line{};
    while (model_file.good()) {
        // read the file line by line
        // -> since the svm_type SHOULD be the first model file entry that should be faster than reading the whole file using the plssvm::detail::io::file_reader
        std::getline(model_file, line);
        detail::to_lower_case(line);

        // check if the current line contains the SVM type
        if (detail::starts_with(line, "svm_type")) {
            std::istringstream iss{ line };
            // skip "svm_type"
            iss >> line;

            // read the SVM type
            svm_type svm{};
            iss >> svm;
            return svm;
        }
        if (detail::starts_with(line, "sv")) {
            // read the last line of the header section but didn't find the SVM type yet -> throw an exception
            throw invalid_file_format_exception{ R"(The provided model file is not a valid LIBSVM model file since "svm_type" is missing!)" };
        }
    }
    // read to the end of the file without finding svm_type or SV -> throw an exception
    throw invalid_file_format_exception{ R"(The provided model file is not a valid LIBSVM model file since "svm_type" AND "SV" are missing!)" };
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
