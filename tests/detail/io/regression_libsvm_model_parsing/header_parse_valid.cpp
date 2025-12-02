/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for parsing an invalid LIBSVM regression model file header.
 */

#include "plssvm/constants.hpp"                                  // plssvm::real_type
#include "plssvm/detail/io/file_reader.hpp"                      // plssvm::detail::io::file_reader
#include "plssvm/detail/io/regression_libsvm_model_parsing.hpp"  // functions to test
#include "plssvm/kernel_function_types.hpp"                      // plssvm::kernel_function_type

#include "tests/custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_EQ
#include "tests/naming.hpp"              // naming::parameter_definition_to_name
#include "tests/types_to_test.hpp"       // util::regression_label_type_kernel_function_type_gtest
#include "tests/utility.hpp"             // util::{temporary_file, instantiate_template_file}

#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, ASSERT_EQ, ::testing::Test

#include <cstddef>  // std::size_t
#include <variant>  // std::get

template <typename T>
class LIBSVMRegressionModelHeaderParseValid : public ::testing::Test {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::kernel_function_type fixture_kernel = util::test_parameter_value_at_v<0, T>;

    /**
     * @brief Check whether the degree field should be read from the file depending on the current kernel type.
     * @return `true` if the degree field should be read, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr bool has_degree() const noexcept {
        return fixture_kernel == plssvm::kernel_function_type::polynomial;
    }

    /**
     * @brief Check whether the gamma field should be read from the file depending on the current kernel type.
     * @return `true` if the gamma field should be read, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr bool has_gamma() const noexcept {
        return fixture_kernel != plssvm::kernel_function_type::linear;
    }

    /**
     * @brief Check whether the coef0 field should be read from the file depending on the current kernel type.
     * @return `true` if the coef0 field should be read, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr bool has_coef0() const noexcept {
        return fixture_kernel == plssvm::kernel_function_type::polynomial || fixture_kernel == plssvm::kernel_function_type::sigmoid;
    }

    /**
     * @brief Return the correct number of header lines depending on the current kernel type.
     * @return the number of header entries expected to be read (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr std::size_t correct_num_header_lines() const noexcept {
        switch (fixture_kernel) {
            case plssvm::kernel_function_type::linear:
                return 6;
            case plssvm::kernel_function_type::polynomial:
                return 9;
            case plssvm::kernel_function_type::rbf:
            case plssvm::kernel_function_type::laplacian:
            case plssvm::kernel_function_type::chi_squared:
                return 7;
            case plssvm::kernel_function_type::sigmoid:
                return 8;
        }
        return 0;
    }
};

TYPED_TEST_SUITE(LIBSVMRegressionModelHeaderParseValid, util::regression_label_type_kernel_function_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMRegressionModelHeaderParseValid, Read) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // create temporary file
    const util::temporary_file template_file{};
    util::instantiate_template_file<label_type>(PLSSVM_TEST_PATH "/data/model/regression/6x4_TEMPLATE.libsvm.model", template_file.filename, kernel);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, num_header_lines] = plssvm::detail::io::parse_libsvm_model_header_regression(reader.lines());

    // check for correctness

    // check parameter
    EXPECT_EQ(params.kernel_type, kernel);
    if (this->has_degree()) {
        EXPECT_EQ(params.degree, 2);
    }
    if (this->has_gamma()) {
        EXPECT_EQ(std::get<plssvm::real_type>(params.gamma), plssvm::real_type{ 0.25 });
    }
    if (this->has_coef0()) {
        EXPECT_EQ(params.coef0, plssvm::real_type{ 1.5 });
    }

    // check rho values
    ASSERT_EQ(rho.size(), 1);
    EXPECT_FLOATING_POINT_EQ(rho.front(), plssvm::real_type{ 0.32260160011873423 });

    // check number of header lines
    EXPECT_EQ(num_header_lines, this->correct_num_header_lines());
}
