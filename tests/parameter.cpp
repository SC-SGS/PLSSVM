/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the parameter class encapsulating all important SVM parameter.
 */

#include "plssvm/parameter.hpp"

#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl::kernel_invocation_type

#include "custom_test_macros.hpp"                           // EXPECT_CONVERSION_TO_STRING, EXPECT_FLOATING_POINT_EQ
#include "naming.hpp"                                       // naming::real_type_to_name
#include "types_to_test.hpp"                                // util::real_type_gtest

#include "fmt/core.h"                                       // fmt::format

#include "gtest/gtest.h"                                    // TYPED_TEST, TYPED_TEST_SUITE, TEST, EXPECT_EQ, EXPECT_FLOAT_EQ, EXPECT_DOUBLE_EQ, EXPECT_TRUE, EXPECT_FALSE
                                                            // ::testing::{Test, Types}

template <typename T>
class Parameter : public ::testing::Test {};

// testsuite for "normal" tests
TYPED_TEST_SUITE(Parameter, util::real_type_gtest, naming::real_type_to_name);

TYPED_TEST(Parameter, default_construct) {
    using real_type = TypeParam;
    // default construct parameter set
    const plssvm::detail::parameter<real_type> param{};

    // test default values
    EXPECT_TRUE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::linear);
    EXPECT_TRUE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 3);
    EXPECT_TRUE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), real_type{ 0.0 });
    EXPECT_TRUE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), real_type{ 0.0 });
    EXPECT_TRUE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), real_type{ 1.0 });
}
TYPED_TEST(Parameter, construct) {
    using real_type = TypeParam;
    // construct a parameter set explicitly overwriting the default values
    const plssvm::detail::parameter<real_type> param{ plssvm::kernel_function_type::polynomial, 1, -1.0, 2.5, 0.05 };

    // test default values
    EXPECT_FALSE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(param.kernel_type.get_default(), plssvm::kernel_function_type::linear);

    EXPECT_FALSE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 1);
    EXPECT_EQ(param.degree.get_default(), 3);

    EXPECT_FALSE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(param.gamma.get_default(), real_type{ 0.0 });

    EXPECT_FALSE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), real_type{ 2.5 });
    EXPECT_FLOATING_POINT_EQ(param.coef0.get_default(), real_type{ 0.0 });

    EXPECT_FALSE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), real_type{ 0.05 });
    EXPECT_FLOATING_POINT_EQ(param.cost.get_default(), real_type{ 1.0 });
}
TYPED_TEST(Parameter, construct_named_args_all) {
    using real_type = TypeParam;
    // construct a parameter set explicitly overwriting the default values using named parameters
    const plssvm::detail::parameter<real_type> param{
        plssvm::kernel_type = plssvm::kernel_function_type::polynomial,
        plssvm::degree = 1,
        plssvm::gamma = -1.0,
        plssvm::coef0 = 2.5,
        plssvm::cost = 0.05
    };

    // test default values
    EXPECT_FALSE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(param.kernel_type.get_default(), plssvm::kernel_function_type::linear);

    EXPECT_FALSE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 1);
    EXPECT_EQ(param.degree.get_default(), 3);

    EXPECT_FALSE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(param.gamma.get_default(), real_type{ 0.0 });

    EXPECT_FALSE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), real_type{ 2.5 });
    EXPECT_FLOATING_POINT_EQ(param.coef0.get_default(), real_type{ 0.0 });

    EXPECT_FALSE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), real_type{ 0.05 });
    EXPECT_FLOATING_POINT_EQ(param.cost.get_default(), real_type{ 1.0 });
}
TYPED_TEST(Parameter, construct_named_args) {
    using real_type = TypeParam;
    // construct a parameter set explicitly overwriting some default values using named parameters
    const plssvm::detail::parameter<real_type> param{
        plssvm::kernel_type = plssvm::kernel_function_type::polynomial,
        plssvm::cost = 0.05,
        plssvm::gamma = -1.0
    };

    // test default values
    EXPECT_FALSE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(param.kernel_type.get_default(), plssvm::kernel_function_type::linear);

    EXPECT_TRUE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 3);
    EXPECT_EQ(param.degree.get_default(), 3);

    EXPECT_FALSE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(param.gamma.get_default(), real_type{ 0.0 });

    EXPECT_TRUE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), real_type{ 0.0 });
    EXPECT_FLOATING_POINT_EQ(param.coef0.get_default(), real_type{ 0.0 });

    EXPECT_FALSE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), real_type{ 0.05 });
    EXPECT_FLOATING_POINT_EQ(param.cost.get_default(), real_type{ 1.0 });
}
TYPED_TEST(Parameter, construct_parameter_and_named_args) {
    using real_type = TypeParam;
    // construct a parameter set
    const plssvm::detail::parameter<real_type> param_base{
        plssvm::kernel_type = plssvm::kernel_function_type::polynomial,
        plssvm::cost = 0.05,
        plssvm::gamma = -1.0
    };

    // create new parameter set using a previous parameter set together with some named parameters
    const plssvm::detail::parameter<real_type> param{ param_base,
                                                      plssvm::kernel_type = plssvm::kernel_function_type::rbf,
                                                      plssvm::sycl_implementation_type = plssvm::sycl::implementation_type::hipsycl,
                                                      plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::hierarchical };

    // test default values
    EXPECT_FALSE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::rbf);
    EXPECT_EQ(param.kernel_type.get_default(), plssvm::kernel_function_type::linear);

    EXPECT_TRUE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 3);
    EXPECT_EQ(param.degree.get_default(), 3);

    EXPECT_FALSE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(param.gamma.get_default(), real_type{ 0.0 });

    EXPECT_TRUE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), real_type{ 0.0 });
    EXPECT_FLOATING_POINT_EQ(param.coef0.get_default(), real_type{ 0.0 });

    EXPECT_FALSE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), real_type{ 0.05 });
    EXPECT_FLOATING_POINT_EQ(param.cost.get_default(), real_type{ 1.0 });
}

TEST(Parameter, conversion_double_to_float) {
    const plssvm::detail::parameter<double> from{};
    // cast from double to float
    const plssvm::detail::parameter<float> to{ static_cast<plssvm::detail::parameter<float>>(from) };

    // nothing should have changed
    EXPECT_EQ(from.kernel_type, to.kernel_type);
    EXPECT_EQ(from.degree, to.degree);
    EXPECT_FLOAT_EQ(static_cast<float>(from.gamma.value()), to.gamma.value());
    EXPECT_FLOAT_EQ(static_cast<float>(from.coef0.value()), to.coef0.value());
    EXPECT_FLOAT_EQ(static_cast<float>(from.cost.value()), to.cost.value());
}
TEST(Parameter, conversion_float_to_double) {
    const plssvm::detail::parameter<float> from{};
    // cast from float to double
    const plssvm::detail::parameter<double> to{ static_cast<plssvm::detail::parameter<double>>(from) };

    // nothing should have changed
    EXPECT_EQ(from.kernel_type, to.kernel_type);
    EXPECT_EQ(from.degree, to.degree);
    EXPECT_DOUBLE_EQ(static_cast<double>(from.gamma.value()), to.gamma.value());
    EXPECT_DOUBLE_EQ(static_cast<double>(from.coef0.value()), to.coef0.value());
    EXPECT_DOUBLE_EQ(static_cast<double>(from.cost.value()), to.cost.value());
}
TYPED_TEST(Parameter, conversion_same_type) {
    using real_type = TypeParam;
    const plssvm::detail::parameter<real_type> from{};
    // cast from one parameter set to another OF THE SAME TYPE
    const plssvm::detail::parameter<real_type> to{ static_cast<plssvm::detail::parameter<real_type>>(from) };

    // nothing should have changed
    EXPECT_EQ(from.kernel_type, to.kernel_type);
    EXPECT_EQ(from.degree, to.degree);
    EXPECT_FLOATING_POINT_EQ(from.gamma.value(), to.gamma.value());
    EXPECT_FLOATING_POINT_EQ(from.coef0.value(), to.coef0.value());
    EXPECT_FLOATING_POINT_EQ(from.cost.value(), to.cost.value());
}

TYPED_TEST(Parameter, equal) {
    using real_type = TypeParam;
    // test whether different parameter sets are equal, i.e., all member variables have the same value
    const plssvm::detail::parameter<real_type> params1{ plssvm::kernel_function_type::rbf, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params2{ plssvm::kernel_function_type::rbf, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params3{ plssvm::kernel_function_type::linear, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params4{ plssvm::kernel_function_type::rbf, 2, 0.02, 1.5, 1.0 };

    // test
    EXPECT_TRUE(params1 == params2);
    EXPECT_FALSE(params1 == params3);
    EXPECT_FALSE(params1 == params4);
    EXPECT_FALSE(params3 == params4);
}
TYPED_TEST(Parameter, equal_default_constructed) {
    using real_type = TypeParam;
    // test whether two default constructed parameter sets are equal, i.e., all member variables have the same value
    const plssvm::detail::parameter<real_type> params1{};
    const plssvm::detail::parameter<real_type> params2{};

    // two default constructed parameter sets must be equal
    EXPECT_TRUE(params1 == params2);
}

TYPED_TEST(Parameter, unequal) {
    using real_type = TypeParam;
    // test whether different parameter sets are unequal, i.e., any member variables differ in value
    const plssvm::detail::parameter<real_type> params1{ plssvm::kernel_function_type::rbf, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params2{ plssvm::kernel_function_type::rbf, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params3{ plssvm::kernel_function_type::linear, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params4{ plssvm::kernel_function_type::rbf, 2, 0.02, 1.5, 1.0 };

    // test
    EXPECT_FALSE(params1 != params2);
    EXPECT_TRUE(params1 != params3);
    EXPECT_TRUE(params1 != params4);
    EXPECT_TRUE(params3 != params4);
}
TYPED_TEST(Parameter, unequal_default_constructed) {
    using real_type = TypeParam;
    // test whether two default constructed parameter sets are unequal, i.e., any member variables differ in value
    const plssvm::detail::parameter<real_type> params1{};
    const plssvm::detail::parameter<real_type> params2{};

    // two default constructed parameter sets must be equal
    EXPECT_FALSE(params1 != params2);
}

TYPED_TEST(Parameter, equivalent_member_function) {
    using real_type = TypeParam;
    // test whether different parameter sets are equivalent, i.e., all member variables IMPORTANT FOR THE KERNEL TYPE have the same value
    const plssvm::detail::parameter<real_type> params1{ plssvm::kernel_function_type::rbf, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params2{ plssvm::kernel_function_type::rbf, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params3{ plssvm::kernel_function_type::linear, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params4{ plssvm::kernel_function_type::rbf, 2, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params5{ plssvm::kernel_function_type::linear, 2, -0.02, 0.5, 1.0 };
    const plssvm::detail::parameter<real_type> params6{ plssvm::kernel_function_type::polynomial, 2, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params7{ plssvm::kernel_function_type::polynomial, 2, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params8{ static_cast<plssvm::kernel_function_type>(3), 3, 0.2, -1.5, 0.1 };
    const plssvm::detail::parameter<real_type> params9{ static_cast<plssvm::kernel_function_type>(3), 3, 0.2, -1.5, 0.1 };

    // test
    EXPECT_TRUE(params1.equivalent(params2));
    EXPECT_FALSE(params1.equivalent(params3));
    EXPECT_TRUE(params1.equivalent(params4));  // only differ in degree, which is unimportant for the rbf kernel -> still equivalent
    EXPECT_FALSE(params3.equivalent(params4));
    EXPECT_TRUE(params3.equivalent(params5));
    EXPECT_TRUE(params6.equivalent(params7));
    EXPECT_FALSE(params6.equivalent(params8));
    EXPECT_FALSE(params8.equivalent(params9));
}
TYPED_TEST(Parameter, equivalent_member_function_default_constructed) {
    using real_type = TypeParam;
    // test whether two default constructed parameter sets are equal, i.e., all member variables have the same value
    const plssvm::detail::parameter<real_type> params1{};
    const plssvm::detail::parameter<real_type> params2{};

    // two default constructed parameter sets must be equal
    EXPECT_TRUE(params1.equivalent(params2));
}

TYPED_TEST(Parameter, equivalent_free_function) {
    using real_type = TypeParam;
    // test whether different parameter sets are equivalent, i.e., all member variables IMPORTANT FOR THE KERNEL TYPE have the same value
    const plssvm::detail::parameter<real_type> params1{ plssvm::kernel_function_type::rbf, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params2{ plssvm::kernel_function_type::rbf, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params3{ plssvm::kernel_function_type::linear, 3, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params4{ plssvm::kernel_function_type::rbf, 2, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params5{ plssvm::kernel_function_type::linear, 2, -0.02, 0.5, 1.0 };
    const plssvm::detail::parameter<real_type> params6{ plssvm::kernel_function_type::polynomial, 2, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params7{ plssvm::kernel_function_type::polynomial, 2, 0.02, 1.5, 1.0 };
    const plssvm::detail::parameter<real_type> params8{ static_cast<plssvm::kernel_function_type>(3), 3, 0.2, -1.5, 0.1 };
    const plssvm::detail::parameter<real_type> params9{ static_cast<plssvm::kernel_function_type>(3), 3, 0.2, -1.5, 0.1 };

    // test
    EXPECT_TRUE(plssvm::detail::equivalent(params1, params2));
    EXPECT_FALSE(plssvm::detail::equivalent(params1, params3));
    EXPECT_TRUE(plssvm::detail::equivalent(params1, params4));  // only differ in degree, which is unimportant for the rbf kernel -> still equivalent
    EXPECT_FALSE(plssvm::detail::equivalent(params3, params4));
    EXPECT_TRUE(plssvm::detail::equivalent(params3, params5));
    EXPECT_TRUE(plssvm::detail::equivalent(params6, params7));
    EXPECT_FALSE(plssvm::detail::equivalent(params6, params8));
    EXPECT_FALSE(plssvm::detail::equivalent(params8, params9));
}
TYPED_TEST(Parameter, equivalent_free_function_default_constructed) {
    using real_type = TypeParam;
    // test whether two default constructed parameter sets are equal, i.e., all member variables have the same value
    const plssvm::detail::parameter<real_type> params1{};
    const plssvm::detail::parameter<real_type> params2{};

    // two default constructed parameter sets must be equal
    EXPECT_TRUE(plssvm::detail::equivalent(params1, params2));
}

TYPED_TEST(Parameter, to_string) {
    using real_type = TypeParam;
    // check conversions to std::string
    const plssvm::detail::parameter<real_type> param{ plssvm::kernel_function_type::linear, 3, 0.0, 0.0, 1.0 };
    EXPECT_CONVERSION_TO_STRING(param, fmt::format("kernel_type                 linear\n"
                                                   "degree                      3\n"
                                                   "gamma                       0\n"
                                                   "coef0                       0\n"
                                                   "cost                        1\n"
                                                   "real_type                   {}\n",
                                                   plssvm::detail::arithmetic_type_name<real_type>()));
}