/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*
* @brief Perform DPC++ compiler check.
*/

#include "sycl/sycl.hpp"

int main() {
    [[maybe_unused]] const auto version = __SYCL_COMPILER_VERSION;
    return 0;
}