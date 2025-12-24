/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Convert a provided type to the corresponding MPI_Datatype, if possible.
 */

#ifndef PLSSVM_MPI_DETAIL_MPI_DATATYPE_HPP_
#define PLSSVM_MPI_DETAIL_MPI_DATATYPE_HPP_
#pragma once

#if defined(PLSSVM_HAS_MPI_ENABLED)

    #include "mpi.h"  // MPI_Datatype, various MPI datatypes

    #include <complex>      // std::complex
    #include <type_traits>  // std::enable_if_t, std::is_enum_v, std::underlying_type_t

    /**
     * @def PLSSVM_CREATE_MPI_DATATYPE_MAPPING
     * @brief Defines a macro to create all possible conversion from a C++ type to a MPI_Datatype.
     * @param[in] cpp_type the C++ type
     * @param[in] mpi_type the corresponding MPI_Datatype
     */
    #define PLSSVM_CREATE_MPI_DATATYPE_MAPPING(cpp_type, mpi_type) \
        template <>                                                \
        [[nodiscard]] inline MPI_Datatype mpi_datatype<cpp_type>() { return mpi_type; }

namespace plssvm::mpi::detail {

/**
 * @brief Tries to convert the given C++ type to its corresponding MPI_Datatype.
 * @details The definition is marked as **deleted** if `T` isn't representable as [`MPI_Datatype`](https://www.mpi-forum.org/docs/mpi-2.2/mpi22-report/node44.htm) or an enum.
 * @tparam T the type to convert to a MPI_Datatype
 * @return the corresponding MPI_Datatype (`[[nodiscard]]`)
 */
template <typename T, std::enable_if_t<!std::is_enum_v<T>, bool> = true>
[[nodiscard]] MPI_Datatype mpi_datatype() = delete;

PLSSVM_CREATE_MPI_DATATYPE_MAPPING(bool, MPI_C_BOOL)

// character types
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(char, MPI_CHAR)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(signed char, MPI_SIGNED_CHAR)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(unsigned char, MPI_UNSIGNED_CHAR)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(wchar_t, MPI_WCHAR)

// integer types
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(signed short, MPI_SHORT)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(unsigned short, MPI_UNSIGNED_SHORT)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(signed int, MPI_INT)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(unsigned int, MPI_UNSIGNED)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(signed long int, MPI_LONG)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(unsigned long int, MPI_UNSIGNED_LONG)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(signed long long int, MPI_LONG_LONG)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(unsigned long long int, MPI_UNSIGNED_LONG_LONG)
// PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::int8_t, MPI_INT8_T)
// PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::int16_t, MPI_INT16_T)
// PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::int32_t, MPI_INT32_T)
// PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::int64_t, MPI_INT64_T)
// PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::uint8_t, MPI_UINT8_T)
// PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::uint16_t, MPI_UINT16_T)
// PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::uint32_t, MPI_UINT32_T)
// PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::uint64_t, MPI_UINT64_T)

// floating point types
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(float, MPI_FLOAT)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(double, MPI_DOUBLE)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(long double, MPI_LONG_DOUBLE)

// complex types
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::complex<float>, MPI_C_COMPLEX)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::complex<double>, MPI_C_DOUBLE_COMPLEX)
PLSSVM_CREATE_MPI_DATATYPE_MAPPING(std::complex<long double>, MPI_C_LONG_DOUBLE_COMPLEX)

/**
 * @brief Specialization for enums: for enums, use their underlying type in MPI communications.
 * @tparam T the enum type to convert to a MPI_Datatype
 * @return the corresponding MPI_Datatype (`[[nodiscard]]`)
 */
template <typename T, std::enable_if_t<std::is_enum_v<T>, bool> = true>
[[nodiscard]] inline MPI_Datatype mpi_datatype() {
    return mpi_datatype<std::underlying_type_t<T>>();
}

}  // namespace plssvm::mpi::detail

    #undef PLSSVM_CREATE_MPI_DATATYPE_MAPPING

#endif

#endif  // PLSSVM_MPI_DETAIL_MPI_DATATYPE_HPP_
