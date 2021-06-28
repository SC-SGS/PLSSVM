// #include "CSVM.hpp"
#include "mocks/CSVM.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#include <string>
#include <unistd.h>

// Demonstrate some basic assertions.
TEST(HelloTest, BasicAssertions) {
    // Expect two strings not to be equal.
    EXPECT_STRNE("hello", "world");
    // Expect equality.
    EXPECT_EQ(7 * 6, 42);
}

TEST(IO, libsvmFormat) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    csvm.libsvmParser("data/5x4.libsvm"); //TODO: add comments etc to libsvm test file
    EXPECT_EQ(csvm.get_num_data_points(), 5);
    EXPECT_EQ(csvm.get_num_features(), 4);

    std::vector<std::vector<real_t>> expected{
        {-1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288},
        {-0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026},
        {0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387},
        {-0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827},
        {1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514},
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(IO, arffFormat) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    csvm.arffParser("data/5x4.arff"); //TODO: add comments etc to arff test file
    EXPECT_EQ(csvm.get_num_data_points(), 5);
    EXPECT_EQ(csvm.get_num_features(), 4);
    std::vector<std::vector<real_t>> expected{
        {-1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288},
        {-0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026},
        {0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387},
        {-0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827},
        {1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514},
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(IO, libsvmFormatIllFormed) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    EXPECT_ANY_THROW(csvm.libsvmParser("data/5x5.arff");); //TODO: change to EXPECT_THROW(statement,exception_type) if exception is implemented
}

TEST(IO, arffFormatIllFormed) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    EXPECT_ANY_THROW(csvm.arffParser("data/5x5.libsvm");); //TODO: change to EXPECT_THROW(statement,exception_type) if exception is implemented
}

TEST(IO, libsvmNoneExistingFile) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    EXPECT_ANY_THROW(csvm.libsvmParser("data/5x5.ar");); //TODO: change to EXPECT_THROW(statement,exception_type) if exception is implemented
}

TEST(IO, arffNoneExistingFile) {
    MockCSVM csvm(1., 1., 0, 1., 1., 1., false);
    EXPECT_ANY_THROW(csvm.arffParser("data/5x5.lib");); //TODO: change to EXPECT_THROW(statement,exception_type) if exception is implemented
}