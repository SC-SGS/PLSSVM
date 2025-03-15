/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for MPI data type mapper functions.
 * @note Assumes only a single MPI rank, since more are **not** supported in our tests!
 */

#include "plssvm/mpi/detail/mpi_datatype.hpp"

#include "gtest/gtest.h"  // TEST, EXPECT_FALSE, EXPECT_DEATH

#include <complex>  // std::complex

#if defined(PLSSVM_HAS_MPI_ENABLED)

TEST(MPIDataTypes, mpi_datatype) {
    // check type conversions
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<bool>(), MPI_C_BOOL);

    // character types
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<char>(), MPI_CHAR);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<signed char>(), MPI_SIGNED_CHAR);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<unsigned char>(), MPI_UNSIGNED_CHAR);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<wchar_t>(), MPI_WCHAR);

    // integer types
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<signed short>(), MPI_SHORT);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<unsigned short>(), MPI_UNSIGNED_SHORT);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<signed int>(), MPI_INT);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<unsigned int>(), MPI_UNSIGNED);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<signed long int>(), MPI_LONG);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<unsigned long int>(), MPI_UNSIGNED_LONG);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<signed long long int>(), MPI_LONG_LONG);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<unsigned long long int>(), MPI_UNSIGNED_LONG_LONG);

    // floating point types
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<float>(), MPI_FLOAT);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<double>(), MPI_DOUBLE);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<long double>(), MPI_LONG_DOUBLE);

    // complex types
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<std::complex<float>>(), MPI_C_COMPLEX);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<std::complex<double>>(), MPI_C_DOUBLE_COMPLEX);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<std::complex<long double>>(), MPI_C_LONG_DOUBLE_COMPLEX);
}

enum class dummy1 : int { };
enum class dummy2 : char { };

TEST(MPIDataTypes, mpi_datatype_from_enum) {
    // check type conversions from enum's underlying type
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<dummy1>(), MPI_INT);
    EXPECT_EQ(plssvm::mpi::detail::mpi_datatype<dummy2>(), MPI_CHAR);
}

#endif
