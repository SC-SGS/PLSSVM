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
#include "plssvm/detail/arithmetic_type_name.hpp"           // plssvm::detail::arithmetic_type_name

#include "custom_test_macros.hpp"  // EXPECT_CONVERSION_TO_STRING, EXPECT_FLOATING_POINT_EQ

#include "fmt/core.h"  // fmt::format

#include "gtest/gtest.h"  // TEST, EXPECT_EQ, EXPECT_FLOAT_EQ, EXPECT_DOUBLE_EQ, EXPECT_TRUE, EXPECT_FALSE

TEST(Parameter, default_construct) {
    // default construct parameter set
    const plssvm::parameter param{};

    // test default values
    EXPECT_TRUE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::rbf);
    EXPECT_TRUE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 3);
    EXPECT_TRUE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), plssvm::real_type{ 0.0 });
    EXPECT_TRUE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), plssvm::real_type{ 0.0 });
    EXPECT_TRUE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), plssvm::real_type{ 1.0 });
}
TEST(Parameter, construct) {
    // construct a parameter set explicitly overwriting the default values
    const plssvm::parameter param{ plssvm::kernel_function_type::polynomial, 1, plssvm::real_type{ -1.0 }, plssvm::real_type{ 2.5 }, plssvm::real_type{ 0.05 } };

    // test default values
    EXPECT_FALSE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(param.kernel_type.get_default(), plssvm::kernel_function_type::rbf);

    EXPECT_FALSE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 1);
    EXPECT_EQ(param.degree.get_default(), 3);

    EXPECT_FALSE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), plssvm::real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(param.gamma.get_default(), plssvm::real_type{ 0.0 });

    EXPECT_FALSE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), plssvm::real_type{ 2.5 });
    EXPECT_FLOATING_POINT_EQ(param.coef0.get_default(), plssvm::real_type{ 0.0 });

    EXPECT_FALSE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), plssvm::real_type{ 0.05 });
    EXPECT_FLOATING_POINT_EQ(param.cost.get_default(), plssvm::real_type{ 1.0 });
}
TEST(Parameter, construct_named_args_all) {
    // construct a parameter set explicitly overwriting all default values using named parameters
    const plssvm::parameter param{
        plssvm::kernel_type = plssvm::kernel_function_type::polynomial,
        plssvm::degree = 1,
        plssvm::gamma = -1.0,
        plssvm::coef0 = 2.5,
        plssvm::cost = 0.05
    };

    // test default values
    EXPECT_FALSE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(param.kernel_type.get_default(), plssvm::kernel_function_type::rbf);

    EXPECT_FALSE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 1);
    EXPECT_EQ(param.degree.get_default(), 3);

    EXPECT_FALSE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), plssvm::real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(param.gamma.get_default(), plssvm::real_type{ 0.0 });

    EXPECT_FALSE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), plssvm::real_type{ 2.5 });
    EXPECT_FLOATING_POINT_EQ(param.coef0.get_default(), plssvm::real_type{ 0.0 });

    EXPECT_FALSE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), plssvm::real_type{ 0.05 });
    EXPECT_FLOATING_POINT_EQ(param.cost.get_default(), plssvm::real_type{ 1.0 });
}
TEST(Parameter, construct_named_args) {
    // construct a parameter set explicitly overwriting some default values using named parameters
    const plssvm::parameter param{
        plssvm::kernel_type = plssvm::kernel_function_type::polynomial,
        plssvm::cost = 0.05,
        plssvm::gamma = -1.0
    };

    // test default values
    EXPECT_FALSE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::polynomial);
    EXPECT_EQ(param.kernel_type.get_default(), plssvm::kernel_function_type::rbf);

    EXPECT_TRUE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 3);
    EXPECT_EQ(param.degree.get_default(), 3);

    EXPECT_FALSE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), plssvm::real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(param.gamma.get_default(), plssvm::real_type{ 0.0 });

    EXPECT_TRUE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), plssvm::real_type{ 0.0 });
    EXPECT_FLOATING_POINT_EQ(param.coef0.get_default(), plssvm::real_type{ 0.0 });

    EXPECT_FALSE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), plssvm::real_type{ 0.05 });
    EXPECT_FLOATING_POINT_EQ(param.cost.get_default(), plssvm::real_type{ 1.0 });
}
TEST(Parameter, construct_parameter_and_named_args) {
    // construct a parameter set
    const plssvm::parameter param_base{
        plssvm::kernel_type = plssvm::kernel_function_type::polynomial,
        plssvm::cost = 0.05,
        plssvm::gamma = -1.0
    };

    // create new parameter set using a previous parameter set together with some named parameters
    const plssvm::parameter param{ param_base,
                                   plssvm::kernel_type = plssvm::kernel_function_type::rbf,
                                   plssvm::sycl_implementation_type = plssvm::sycl::implementation_type::adaptivecpp,
                                   plssvm::sycl_kernel_invocation_type = plssvm::sycl::kernel_invocation_type::nd_range };

    // test default values
    EXPECT_FALSE(param.kernel_type.is_default());
    EXPECT_EQ(param.kernel_type.value(), plssvm::kernel_function_type::rbf);
    EXPECT_EQ(param.kernel_type.get_default(), plssvm::kernel_function_type::rbf);

    EXPECT_TRUE(param.degree.is_default());
    EXPECT_EQ(param.degree.value(), 3);
    EXPECT_EQ(param.degree.get_default(), 3);

    EXPECT_FALSE(param.gamma.is_default());
    EXPECT_FLOATING_POINT_EQ(param.gamma.value(), plssvm::real_type{ -1.0 });
    EXPECT_FLOATING_POINT_EQ(param.gamma.get_default(), plssvm::real_type{ 0.0 });

    EXPECT_TRUE(param.coef0.is_default());
    EXPECT_FLOATING_POINT_EQ(param.coef0.value(), plssvm::real_type{ 0.0 });
    EXPECT_FLOATING_POINT_EQ(param.coef0.get_default(), plssvm::real_type{ 0.0 });

    EXPECT_FALSE(param.cost.is_default());
    EXPECT_FLOATING_POINT_EQ(param.cost.value(), plssvm::real_type{ 0.05 });
    EXPECT_FLOATING_POINT_EQ(param.cost.get_default(), plssvm::real_type{ 1.0 });
}

TEST(Parameter, equal) {
    // test whether different parameter sets are equal, i.e., all member variables have the same value
    const plssvm::parameter params1{ plssvm::kernel_function_type::rbf, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params2{ plssvm::kernel_function_type::rbf, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params3{ plssvm::kernel_function_type::linear, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params4{ plssvm::kernel_function_type::rbf, 2, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };

    // test
    EXPECT_TRUE(params1 == params2);
    EXPECT_FALSE(params1 == params3);
    EXPECT_FALSE(params1 == params4);
    EXPECT_FALSE(params3 == params4);
}
TEST(Parameter, equal_default_constructed) {
    // test whether two default constructed parameter sets are equal, i.e., all member variables have the same value
    const plssvm::parameter params1{};
    const plssvm::parameter params2{};

    // two default constructed parameter sets must be equal
    EXPECT_TRUE(params1 == params2);
}

TEST(Parameter, unequal) {
    // test whether different parameter sets are unequal, i.e., any member variables differ in value
    const plssvm::parameter params1{ plssvm::kernel_function_type::rbf, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params2{ plssvm::kernel_function_type::rbf, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params3{ plssvm::kernel_function_type::linear, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params4{ plssvm::kernel_function_type::rbf, 2, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };

    // test
    EXPECT_FALSE(params1 != params2);
    EXPECT_TRUE(params1 != params3);
    EXPECT_TRUE(params1 != params4);
    EXPECT_TRUE(params3 != params4);
}
TEST(Parameter, unequal_default_constructed) {
    // test whether two default constructed parameter sets are unequal, i.e., any member variables differ in value
    const plssvm::parameter params1{};
    const plssvm::parameter params2{};

    // two default constructed parameter sets must be equal
    EXPECT_FALSE(params1 != params2);
}

TEST(Parameter, equivalent_member_function) {
    // test whether different parameter sets are equivalent, i.e., all member variables IMPORTANT FOR THE KERNEL TYPE have the same value
    const plssvm::parameter params1{ plssvm::kernel_function_type::rbf, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params2{ plssvm::kernel_function_type::rbf, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params3{ plssvm::kernel_function_type::linear, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params4{ plssvm::kernel_function_type::rbf, 2, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params5{ plssvm::kernel_function_type::linear, 2, plssvm::real_type{ -0.02 }, plssvm::real_type{ 0.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params6{ plssvm::kernel_function_type::polynomial, 2, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params7{ plssvm::kernel_function_type::polynomial, 2, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params8{ static_cast<plssvm::kernel_function_type>(3), 3, plssvm::real_type{ 0.2 }, plssvm::real_type{ -1.5 }, plssvm::real_type{ 0.1 } };
    const plssvm::parameter params9{ static_cast<plssvm::kernel_function_type>(3), 3, plssvm::real_type{ 0.2 }, plssvm::real_type{ -1.5 }, plssvm::real_type{ 0.1 } };

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
TEST(Parameter, equivalent_member_function_default_constructed) {
    // test whether two default constructed parameter sets are equal, i.e., all member variables have the same value
    const plssvm::parameter params1{};
    const plssvm::parameter params2{};

    // two default constructed parameter sets must be equal
    EXPECT_TRUE(params1.equivalent(params2));
}

TEST(Parameter, equivalent_free_function) {
    // test whether different parameter sets are equivalent, i.e., all member variables IMPORTANT FOR THE KERNEL TYPE have the same value
    const plssvm::parameter params1{ plssvm::kernel_function_type::rbf, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params2{ plssvm::kernel_function_type::rbf, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params3{ plssvm::kernel_function_type::linear, 3, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params4{ plssvm::kernel_function_type::rbf, 2, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params5{ plssvm::kernel_function_type::linear, 2, plssvm::real_type{ -0.02 }, plssvm::real_type{ 0.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params6{ plssvm::kernel_function_type::polynomial, 2, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params7{ plssvm::kernel_function_type::polynomial, 2, plssvm::real_type{ 0.02 }, plssvm::real_type{ 1.5 }, plssvm::real_type{ 1.0 } };
    const plssvm::parameter params8{ static_cast<plssvm::kernel_function_type>(3), 3, plssvm::real_type{ 0.2 }, plssvm::real_type{ -1.5 }, plssvm::real_type{ 0.1 } };
    const plssvm::parameter params9{ static_cast<plssvm::kernel_function_type>(3), 3, plssvm::real_type{ 0.2 }, plssvm::real_type{ -1.5 }, plssvm::real_type{ 0.1 } };


    // test
    EXPECT_TRUE(plssvm::equivalent(params1, params2));
    EXPECT_FALSE(plssvm::equivalent(params1, params3));
    EXPECT_TRUE(plssvm::equivalent(params1, params4));  // only differ in degree, which is unimportant for the rbf kernel -> still equivalent
    EXPECT_FALSE(plssvm::equivalent(params3, params4));
    EXPECT_TRUE(plssvm::equivalent(params3, params5));
    EXPECT_TRUE(plssvm::equivalent(params6, params7));
    EXPECT_FALSE(plssvm::equivalent(params6, params8));
    EXPECT_FALSE(plssvm::equivalent(params8, params9));
}
TEST(Parameter, equivalent_free_function_default_constructed) {
    // test whether two default constructed parameter sets are equal, i.e., all member variables have the same value
    const plssvm::parameter params1{};
    const plssvm::parameter params2{};

    // two default constructed parameter sets must be equal
    EXPECT_TRUE(plssvm::equivalent(params1, params2));
}

TEST(Parameter, to_string) {
    // check conversions to std::string
    const plssvm::parameter param{ plssvm::kernel_function_type::linear, 3, plssvm::real_type{ 0.0 }, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 } };
    EXPECT_CONVERSION_TO_STRING(param, fmt::format("kernel_type                 linear\n"
                                                   "degree                      3\n"
                                                   "gamma                       0\n"
                                                   "coef0                       0\n"
                                                   "cost                        1\n"
                                                   "real_type                   {}\n",
                                                   plssvm::detail::arithmetic_type_name<plssvm::real_type>()));
}