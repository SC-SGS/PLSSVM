/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements custom literals regarding byte units..
 */

#ifndef PLSSVM_DETAIL_CUSTOM_LITERALS_HPP_
#define PLSSVM_DETAIL_CUSTOM_LITERALS_HPP_
#pragma once

namespace plssvm::detail::literals {

/// Convert bytes to bytes.
constexpr long double operator""_B(const long double val) { return val; }
/// Convert bytes to bytes.
constexpr unsigned long long operator""_B(const unsigned long long val) { return val; }

//*************************************************************************************************************************************//
//                                                    decimal prefix - long double                                                     //
//*************************************************************************************************************************************//

/// Convert 1 KB to bytes (factor 1'000).
constexpr long double operator""_KB(const long double val) { return val * 1000L; }
/// Convert 1 MB to bytes (factor 1'000'000).
constexpr long double operator""_MB(const long double val) { return val * 1000L * 1000L; }
/// Convert 1 GB to bytes (factor 1'000'000'000).
constexpr long double operator""_GB(const long double val) { return val * 1000L * 1000L * 1000L; }
/// Convert 1 TB to bytes (factor 1'000'000'000'000).
constexpr long double operator""_TB(const long double val) { return val * 1000L * 1000L * 1000L * 1000L; }

//*************************************************************************************************************************************//
//                                                 decimal prefix - unsigned long long                                                 //
//*************************************************************************************************************************************//

/// Convert 1 KB to bytes (factor 1'000).
constexpr unsigned long long operator""_KB(const unsigned long long val) { return val * 1000ULL; }
/// Convert 1 MB to bytes (factor 1'000'000).
constexpr unsigned long long operator""_MB(const unsigned long long val) { return val * 1000ULL * 1000ULL; }
/// Convert 1 GB to bytes (factor 1'000'000'000).
constexpr unsigned long long operator""_GB(const unsigned long long val) { return val * 1000ULL * 1000ULL * 1000ULL; }
/// Convert 1 TB to bytes (factor 1'000'000'000'000).
constexpr unsigned long long operator""_TB(const unsigned long long val) { return val * 1000ULL * 1000ULL * 1000ULL * 1000ULL; }

//*************************************************************************************************************************************//
//                                                     binary prefix - long double                                                     //
//*************************************************************************************************************************************//

/// Convert 1 KiB to bytes (factor 1'024).
constexpr long double operator""_KiB(const long double val) { return val * 1024L; }
/// Convert 1 MiB to bytes (factor 1'048'576).
constexpr long double operator""_MiB(const long double val) { return val * 1024L * 1024L; }
/// Convert 1 GiB to bytes (factor 1'073'741'824).
constexpr long double operator""_GiB(const long double val) { return val * 1024L * 1024L * 1024L; }
/// Convert 1 TiB to bytes (factor 1'099'511'627'776).
constexpr long double operator""_TiB(const long double val) { return val * 1024L * 1024L * 1024L * 1024L; }

//*************************************************************************************************************************************//
//                                                 binary prefix - unsigned long long                                                  //
//*************************************************************************************************************************************//

/// Convert 1 KiB to bytes (factor 1'024).
constexpr unsigned long long operator""_KiB(const unsigned long long val) { return val * 1024ULL; }
/// Convert 1 MiB to bytes (factor 1'048'576).
constexpr unsigned long long operator""_MiB(const unsigned long long val) { return val * 1024ULL * 1024ULL; }
/// Convert 1 GiB to bytes (factor 1'073'741'824).
constexpr unsigned long long operator""_GiB(const unsigned long long val) { return val * 1024ULL * 1024ULL * 1024ULL; }
/// Convert 1 TiB to bytes (factor 1'099'511'627'776).
constexpr unsigned long long operator""_TiB(const unsigned long long val) { return val * 1024ULL * 1024ULL * 1024ULL * 1024ULL; }

}  // namespace plssvm::detail::literals

#endif  // PLSSVM_DETAIL_CUSTOM_LITERALS_HPP_
