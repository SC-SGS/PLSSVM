/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Very basic tests for MPI information functions.
 * @note Assumes only a single MPI rank, since more are **not** supported in our tests!
 */

#include "plssvm/mpi/detail/information.hpp"

#include "plssvm/backend_types.hpp"     // plssvm::backend_type
#include "plssvm/mpi/communicator.hpp"  // plssvm::mpi::communicator
#include "plssvm/solver_types.hpp"      // plssvm::solver_type
#include "plssvm/target_platforms.hpp"  // plssvm::target_platform

#include "tests/utility.hpp"  // util::redirect_output

#include "gtest/gtest.h"  // ::testing::Test, TEST, EXPECT_FALSE

#include <iostream>  // std::cout
#include <string>    // std::string
#include <vector>    // std::vector

class MPIInformation : public ::testing::Test,
                       public util::redirect_output<&std::cout> { };

TEST_F(MPIInformation, gather_and_print_solver_information) {
    // construct an MPI communicator
    const plssvm::mpi::communicator comm{};

    // call the print function
    plssvm::mpi::detail::gather_and_print_solver_information(comm, plssvm::solver_type::cg_explicit);

    // the capture may not be empty
    EXPECT_FALSE(this->get_capture().empty());
}

TEST_F(MPIInformation, gather_and_print_csvm_information_with_device_names) {
    // construct an MPI communicator
    const plssvm::mpi::communicator comm{};

    // call the print function
    plssvm::mpi::detail::gather_and_print_csvm_information(comm, plssvm::backend_type::cuda, plssvm::target_platform::gpu_nvidia, std::vector<std::string>{ "GPU1", "GPU2" });

    // the capture may not be empty
    EXPECT_FALSE(this->get_capture().empty());
}

TEST_F(MPIInformation, gather_and_print_csvm_information) {
    // construct an MPI communicator
    const plssvm::mpi::communicator comm{};

    // call the print function
    plssvm::mpi::detail::gather_and_print_csvm_information(comm, plssvm::backend_type::cuda, plssvm::target_platform::gpu_nvidia);

    // the capture may not be empty
    EXPECT_FALSE(this->get_capture().empty());
}
