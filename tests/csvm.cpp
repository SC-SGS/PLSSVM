/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the base C-SVM functions through its mock class.
 */

#include "plssvm/csvm.hpp"       // plssvm::csvm // TODO: necessary?
#include "mock_csvm.hpp"         // mock_csvm
#include "plssvm/parameter.hpp"  // plssvm::parameter

#include "plssvm/detail/utility.hpp"  // plssvm::detail::contains

#include "naming.hpp"   // naming::arithmetic_types_to_name
#include "utility.hpp"  // util::{convert_to_string, convert_from_string, util::generate_random_vector, gtest_assert_floating_point_near}, EXPECT_THROW_WHAT

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_DEATH

#include <algorithm>  // std::generate
#include <array>      // std::array
#include <cstddef>    // std::size_t
#include <sstream>    // std::istringstream
#include <tuple>      // std::ignore
#include <vector>     // std::vector

template <typename T>
class BaseCSVM : public ::testing::Test {};
TYPED_TEST_SUITE(BaseCSVM, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(BaseCSVM, default_construct_from_parameter) {
    using real_type = TypeParam;

    // create mock_csvm using the default parameter (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{};

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), plssvm::parameter<real_type>{});
}
TYPED_TEST(BaseCSVM, construct_from_parameter) {
    using real_type = TypeParam;

    // create parameter class
    const plssvm::parameter<real_type> params{ plssvm::kernel_type::polynomial, 4, real_type{ 0.2 }, real_type{ 0.1 }, real_type{ 0.01 } };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ params };

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}

TYPED_TEST(BaseCSVM, construct_from_parameter_invalid_kernel_type) {
    using real_type = TypeParam;

    // create parameter class
    plssvm::parameter<real_type> params{};
    params.kernel = static_cast<plssvm::kernel_type>(3);

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid kernel type must throw
    EXPECT_THROW_WHAT(mock_csvm<real_type>{ params }, plssvm::invalid_parameter_exception, "Invalid kernel function 3 given!");
}
TYPED_TEST(BaseCSVM, construct_from_parameter_invalid_gamma) {
    using real_type = TypeParam;

    // create parameter class
    plssvm::parameter<real_type> params{};
    params.kernel = plssvm::kernel_type::polynomial;
    params.gamma = -1.0;

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid value for gamma must throw
    EXPECT_THROW_WHAT(mock_csvm<real_type>{ params }, plssvm::invalid_parameter_exception, "gamma must be greater than 0.0, but is -1!");
}

TYPED_TEST(BaseCSVM, construct_linear_from_named_parameters) {
    using real_type = TypeParam;

    // correct parameters
    plssvm::parameter<real_type> params{};
    params.cost = 2.0;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ plssvm::kernel_type::linear, plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}
TYPED_TEST(BaseCSVM, construct_polynomial_from_named_parameters) {
    using real_type = TypeParam;

    // correct parameters
    const plssvm::parameter<real_type> params{ plssvm::kernel_type::polynomial, 4, 0.1, 1.2, 0.001 };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ plssvm::kernel_type::polynomial,
                                     plssvm::degree = params.degree,
                                     plssvm::gamma = params.gamma,
                                     plssvm::coef0 = params.coef0,
                                     plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}
TYPED_TEST(BaseCSVM, construct_rbf_from_named_parameters) {
    using real_type = TypeParam;

    // correct parameters
    plssvm::parameter<real_type> params{};
    params.kernel = plssvm::kernel_type::rbf;
    params.gamma = 0.00001;
    params.cost = 10.0;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ plssvm::kernel_type::rbf,
                                     plssvm::gamma = params.gamma,
                                     plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}

template <typename T>
class BaseCSVMWarning : public BaseCSVM<T> {
  public:
    void start_capturing() {
        sbuf = std::clog.rdbuf();
        std::clog.rdbuf(buffer.rdbuf());
    }
    void end_capturing() {
        std::clog.rdbuf(sbuf);
        sbuf = nullptr;
    }
    std::string get_capture() {
        return buffer.str();
    }

  private:
    std::stringstream buffer{};
    std::streambuf *sbuf{};
};
TYPED_TEST_SUITE(BaseCSVMWarning, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(BaseCSVMWarning, construct_unused_parameter_warning_degree) {
    using real_type = TypeParam;

    // start capture of std::clog
    this->start_capturing();

    const mock_csvm<real_type> csvm{ plssvm::kernel_type::linear, plssvm::degree = 2 };

    // end capture of std::clog
    this->end_capturing();

    EXPECT_EQ(this->get_capture(), "degree parameter provided, which is not used in the linear kernel (u'*v)!\n");
}
TYPED_TEST(BaseCSVMWarning, construct_unused_parameter_warning_gamma) {
    using real_type = TypeParam;

    // start capture of std::clog
    this->start_capturing();

    const mock_csvm<real_type> csvm{ plssvm::kernel_type::linear, plssvm::gamma = 0.1 };

    // end capture of std::clog
    this->end_capturing();

    EXPECT_EQ(this->get_capture(), "gamma parameter provided, which is not used in the linear kernel (u'*v)!\n");
}
TYPED_TEST(BaseCSVMWarning, construct_unused_parameter_warning_coef0) {
    using real_type = TypeParam;

    // start capture of std::clog
    this->start_capturing();

    const mock_csvm<real_type> csvm{ plssvm::kernel_type::linear, plssvm::coef0 = 0.1 };

    // end capture of std::clog
    this->end_capturing();

    EXPECT_EQ(this->get_capture(), "coef0 parameter provided, which is not used in the linear kernel (u'*v)!\n");
}

TYPED_TEST(BaseCSVM, construct_from_named_parameters_invalid_kernel_type) {
    using real_type = TypeParam;

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid kernel type must throw
    EXPECT_THROW_WHAT(mock_csvm<real_type>{ static_cast<plssvm::kernel_type>(3) },
                      plssvm::invalid_parameter_exception,
                      "Invalid kernel function 3 given!");
}
TYPED_TEST(BaseCSVM, construct_from_named_parameters_invalid_gamma) {
    using real_type = TypeParam;

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid value for gamma must throw
    EXPECT_THROW_WHAT((mock_csvm<real_type>{ plssvm::kernel_type::polynomial, plssvm::gamma = -1.0 }),
                      plssvm::invalid_parameter_exception,
                      "gamma must be greater than 0.0, but is -1!");
}

TYPED_TEST(BaseCSVM, get_params) {
    using real_type = TypeParam;

    // create parameter class
    const plssvm::parameter<real_type> params{ plssvm::kernel_type::polynomial, 4, real_type{ 0.2 }, real_type{ 0.1 }, real_type{ 0.01 } };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm<real_type> csvm{ params };

    // check whether the parameters have been set correctly
    const plssvm::parameter<real_type> csvm_params = csvm.get_params();
    EXPECT_EQ(csvm_params, params);
    EXPECT_TRUE(csvm_params.equivalent(params));
}