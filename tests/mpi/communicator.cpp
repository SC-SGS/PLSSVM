/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for MPI communicator wrapper.
 * @note Assumes only a single MPI rank, since more are **not** supported in our tests!
 */

#include "plssvm/mpi/communicator.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::mpi_exception
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix, plssvm::soa_matrix
#include "plssvm/shape.hpp"                  // plssvm::shape

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_OPTIONAL_EQ
#include "tests/utility.hpp"             // util::generate_random_matrix

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "mpi.h"  // MPI_COMM_WORLD, MPI_IDENT, MPI_Comm_compare, MPI_Comm_dup, MPI_Comm_free
#endif

#include "gmock/gmock.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE

#include <chrono>    // std::chrono::milliseconds
#include <cstddef>   // std::size_t
#include <iostream>  // std::cout, std::endl
#include <string>    // std::string
#include <vector>    // std::vector

TEST(MPICommunicator, DefaultConstruct) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // load-balancing weights should be empty
    EXPECT_FALSE(comm.get_load_balancing_weights().has_value());
}

TEST(MPICommunicator, ConstructWeights) {
    const std::vector<std::size_t> weights = { std::size_t{ 42 } };
    // create an MPI communicator with load-balancing weights
    const plssvm::mpi::communicator comm{ weights };

    // load-balancing weights should be set
    ASSERT_TRUE(comm.get_load_balancing_weights().has_value());
    EXPECT_EQ(comm.get_load_balancing_weights().value(), weights);
}

#if defined(PLSSVM_HAS_MPI_ENABLED)
TEST(MPICommunicator, ConstructMPIComm) {
    // create an MPI communicator wrapping an MPI_Comm
    const plssvm::mpi::communicator comm{ MPI_COMM_WORLD };

    // load-balancing weights should be empty
    EXPECT_FALSE(comm.get_load_balancing_weights().has_value());

    // the wrapped MPI communicator should be equal to MPI_COMM_WORLD
    int result{};
    MPI_Comm_compare(static_cast<MPI_Comm>(comm), MPI_COMM_WORLD, &result);
    EXPECT_EQ(result, MPI_IDENT);
}

TEST(MPICommunicator, ConstructMPICommAndWeights) {
    const std::vector<std::size_t> weights = { std::size_t{ 42 } };
    // create a MPI communicator with load-balancing weights
    const plssvm::mpi::communicator comm{ MPI_COMM_WORLD, weights };

    // load-balancing weights should be set
    ASSERT_TRUE(comm.get_load_balancing_weights().has_value());
    EXPECT_EQ(comm.get_load_balancing_weights().value(), weights);

    // the wrapped MPI communicator should be equal to MPI_COMM_WORLD
    int result{};
    MPI_Comm_compare(static_cast<MPI_Comm>(comm), MPI_COMM_WORLD, &result);
    EXPECT_EQ(result, MPI_IDENT);
}
#endif

TEST(MPICommunicator, Size) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // the size must be 1 since MPI is disabled
    EXPECT_EQ(comm.size(), std::size_t{ 1 });
}

TEST(MPICommunicator, Rank) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // the rank must be 0 since MPI is disabled
    EXPECT_EQ(comm.rank(), std::size_t{ 0 });
}

TEST(MPICommunicator, MainRank) {
    // always 0
    EXPECT_EQ(plssvm::mpi::communicator::main_rank(), std::size_t{ 0 });
}

TEST(MPICommunicator, IsMPIEnabled) {
#if defined(PLSSVM_HAS_MPI_ENABLED)
    EXPECT_TRUE(plssvm::mpi::communicator::is_mpi_enabled());
#else
    EXPECT_FALSE(plssvm::mpi::communicator::is_mpi_enabled());
#endif
}

TEST(MPICommunicator, IsMainRank) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // always true since MPI is disabled
    EXPECT_TRUE(comm.is_main_rank());
}

TEST(MPICommunicator, Serialize) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // check if serialize can be called correctly
    comm.serialize([&]() { std::cout << comm.rank() << std::endl; });
}

TEST(MPICommunicator, Gather) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // since MPI is disabled, a call to gather must return a vector containing one value which is equal to the provided one
    const std::vector<int> res = comm.gather(42);
    EXPECT_EQ(res.size(), std::size_t{ 1 });
    EXPECT_EQ(res.front(), 42);
}

TEST(MPICommunicator, GatherString) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // the string to send
    const std::string msg{ "Hello, World!" };

    // since MPI is disabled, a call to gather must return a vector containing one value which is equal to the provided one
    const std::vector<std::string> res = comm.gather(msg);
    EXPECT_EQ(res.size(), std::size_t{ 1 });
    EXPECT_EQ(res.front(), msg);
}

TEST(MPICommunicator, GatherMilli) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // since MPI is disabled, a call to gather must return a vector containing one value which is equal to the provided one
    const std::vector<std::chrono::milliseconds> res = comm.gather(std::chrono::milliseconds{ 13 });
    EXPECT_EQ(res.size(), std::size_t{ 1 });
    EXPECT_EQ(res.front(), std::chrono::milliseconds{ 13 });
}

TEST(MPICommunicator, Allgather) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // since MPI is disabled, a call to allgather should behave like a call to gather
    const std::vector<int> res = comm.allgather(42);
    EXPECT_EQ(res.size(), std::size_t{ 1 });
    EXPECT_EQ(res.front(), 42);
}

TEST(MPICommunicator, AllreduceInplace) {
    // create a default constructed MPI communicator
    const plssvm::mpi::communicator comm{};

    // since MPI is disabled, a call to allreduce_inplace must return a vector containing one value which is equal to the provided one
    {
        auto matr = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 }, plssvm::shape{ 2, 2 });
        const auto matr_correct = matr;
        comm.allreduce_inplace(matr);
        EXPECT_EQ(matr, matr_correct);
    }
    {
        auto matr = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 4, 4 }, plssvm::shape{ 2, 2 });
        const auto matr_correct = matr;
        comm.allreduce_inplace(matr);
        EXPECT_EQ(matr, matr_correct);
    }
}

TEST(MPICommunicator, SetLoadBalancingWeights) {
    const std::vector<std::size_t> weights = { std::size_t{ 42 } };
    // create default constructed MPI communicator
    plssvm::mpi::communicator comm{};

    // should have no load-balancing weights
    ASSERT_FALSE(comm.get_load_balancing_weights().has_value());

    // set new load-balancing weights
    comm.set_load_balancing_weights(weights);

    // now, there should be load-balancing weights
    ASSERT_TRUE(comm.get_load_balancing_weights().has_value());
    EXPECT_EQ(comm.get_load_balancing_weights(), weights);
}

TEST(MPICommunicator, GetLoadBalancingWeights) {
    const std::vector<std::size_t> weights = { std::size_t{ 42 } };
    // create default constructed MPI communicator
    plssvm::mpi::communicator comm{};

    // should have no load-balancing weights
    ASSERT_FALSE(comm.get_load_balancing_weights().has_value());

    // set new load-balancing weights
    comm.set_load_balancing_weights(weights);

    // now, there should be load-balancing weights
    EXPECT_TRUE(comm.get_load_balancing_weights().has_value());
}

TEST(MPICommunicator, Equal) {
    // create two default constructed MPI communicators
    const plssvm::mpi::communicator comm1{};
    const plssvm::mpi::communicator comm2{};

    // since MPI is disabled, two communicator should always be equal
    EXPECT_TRUE(comm1 == comm2);

#if defined(PLSSVM_HAS_MPI_ENABLED)
    const plssvm::mpi::communicator comm3{ MPI_COMM_WORLD };
    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm4{ duplicated_mpi_comm };

    EXPECT_TRUE(comm1 == comm3);
    EXPECT_FALSE(comm1 == comm4);
    EXPECT_FALSE(comm3 == comm4);

    MPI_Comm_free(&duplicated_mpi_comm);
#endif
}

TEST(MPICommunicator, Unequal) {
    // create two default constructed MPI communicators
    const plssvm::mpi::communicator comm1{};
    const plssvm::mpi::communicator comm2{};

    // since MPI is disabled, two communicator should never be unequal
    EXPECT_FALSE(comm1 != comm2);

#if defined(PLSSVM_HAS_MPI_ENABLED)
    const plssvm::mpi::communicator comm3{ MPI_COMM_WORLD };
    // create a duplicated communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm4{ duplicated_mpi_comm };

    EXPECT_FALSE(comm1 != comm3);
    EXPECT_TRUE(comm1 != comm4);
    EXPECT_TRUE(comm3 != comm4);

    MPI_Comm_free(&duplicated_mpi_comm);
#endif
}

TEST(MPICommunicatorDeathTest, ConstructTooFewWeights) {
    // since MPI is not enabled, we can only pass exactly ONE weight value
    EXPECT_THROW_WHAT_MATCHER(plssvm::mpi::communicator{ std::vector<std::size_t>{} }, plssvm::mpi_exception, ::testing::HasSubstr("The number of load balancing weights (0) must match the number of MPI ranks (1)!"));
}

TEST(MPICommunicatorDeathTest, ConstructTooManyWeights) {
    // since MPI is not enabled, we can only pass exactly ONE weight value
    EXPECT_THROW_WHAT_MATCHER((plssvm::mpi::communicator{ std::vector<std::size_t>{ std::size_t{ 1 }, std::size_t{ 2 } } }),
                              plssvm::mpi_exception,
                              ::testing::HasSubstr("The number of load balancing weights (2) must match the number of MPI ranks (1)!"));
}

TEST(MPICommunicatorDeathTest, SetTooFewLoadBalancingWeights) {
    // create default constructed MPI communicator
    plssvm::mpi::communicator comm{};

    // since MPI is not enabled, we can only pass exactly ONE weight value
    EXPECT_THROW_WHAT_MATCHER(comm.set_load_balancing_weights({}), plssvm::mpi_exception, ::testing::HasSubstr("The number of load balancing weights (0) must match the number of MPI ranks (1)!"));
}

TEST(MPICommunicatorDeathTest, SetTooManyLoadBalancingWeights) {
    // create default constructed MPI communicator
    plssvm::mpi::communicator comm{};

    // since MPI is not enabled, we can only pass exactly ONE weight value
    EXPECT_THROW_WHAT_MATCHER((comm.set_load_balancing_weights(std::vector<std::size_t>{ std::size_t{ 1 }, std::size_t{ 2 } })),
                              plssvm::mpi_exception,
                              ::testing::HasSubstr("The number of load balancing weights (2) must match the number of MPI ranks (1)!"));
}
