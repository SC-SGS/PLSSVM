/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains

#include "omp.h"

#include "fmt/core.h"  // fmt::format

#include <string>         // std::string
#include <unordered_map>  // std::unordered_map

namespace plssvm::openmp::detail {

int get_num_threads() {
    // get the number of used OpenMP threads
    int num_omp_threads = 0;
#pragma omp parallel default(none) shared(num_omp_threads)
    {
#pragma omp master
        num_omp_threads = omp_get_num_threads();
    }
    return num_omp_threads;
}

std::string get_openmp_version() {
    // create version map according to https://stackoverflow.com/questions/1304363/how-to-check-the-version-of-openmp-on-linux
    // clang-format off
    static const std::unordered_map<unsigned, std::string> version_map{
        { 1998'10, "1.0" },
        { 2002'03, "2.0" },
        { 2005'05, "2.5" },
        { 2008'05, "3.0" },
        { 2011'07, "3.1" },
        { 2013'07, "4.0" },
        { 2015'11, "4.5" },
        { 2018'11, "5.0" },
        { 2020'11, "5.1" },
        { 2021'11, "5.2" }
    };
    // clang-format on

    // return sanitized version or plain _OPENMP if the version isn't found
    if (::plssvm::detail::contains(version_map, _OPENMP)) {
        return version_map.at(_OPENMP);
    } else {
        return fmt::format("{}", _OPENMP);
    }
}

}  // namespace plssvm::openmp::detail
