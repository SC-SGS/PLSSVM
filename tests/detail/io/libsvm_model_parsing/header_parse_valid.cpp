/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for parsing an invalid LIBSVM model file header.
 */

#include "plssvm/classification_types.hpp"   // plssvm::classification_type
#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/detail/io/libsvm_model_parsing.hpp"
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type

#include "tests/custom_test_macros.hpp"  // EXPECT_FLOATING_POINT_VECTOR_EQ, EXPECT_FLOATING_POINT_EQ
#include "tests/naming.hpp"              // naming::parameter_definition_to_name
#include "tests/types_to_test.hpp"       // util::label_type_classification_type_gtest
#include "tests/utility.hpp"             // util::{temporary_file, get_num_classes, instantiate_template_file, get_correct_model_file_labels,
                                         // get_distinct_label, get_correct_model_file_num_sv_per_class}

#include "fmt/core.h"     // fmt::format
#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, ASSERT_EQ, FAIL, FAIL, ::testing::Test

#include <cstddef>  // std::size_t
#include <string>   // std::string
#include <variant>  // std::get
#include <vector>   // std::vector

template <typename T>
class LIBSVMModelHeaderParseValid : public ::testing::Test {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::kernel_function_type fixture_kernel = util::test_parameter_value_at_v<0, T>;
    constexpr static plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<1, T>;

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
                return 8;
            case plssvm::kernel_function_type::polynomial:
                return 11;
            case plssvm::kernel_function_type::rbf:
            case plssvm::kernel_function_type::laplacian:
            case plssvm::kernel_function_type::chi_squared:
                return 9;
            case plssvm::kernel_function_type::sigmoid:
                return 10;
        }
        return 0;
    }
};

TYPED_TEST_SUITE(LIBSVMModelHeaderParseValid, util::label_type_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelHeaderParseValid, read) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create temporary file
    const util::temporary_file template_file{};
    const std::size_t num_classes_for_label_type = util::get_num_classes<label_type>();
    const std::string template_file_name = fmt::format(PLSSVM_TEST_PATH "/data/model/6x4_{}_{}_TEMPLATE.libsvm.model", num_classes_for_label_type, classification);
    util::instantiate_template_file<label_type>(template_file_name, template_file.filename, kernel);

    // parse the LIBSVM model file header
    plssvm::detail::io::file_reader reader{ template_file.filename };
    reader.read_lines('#');
    const auto &[params, rho, label, different_classes, num_sv_per_class, num_header_lines] = plssvm::detail::io::parse_libsvm_model_header<label_type>(reader.lines());

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
    const std::vector<plssvm::real_type> all_rhos{
        plssvm::real_type{ 0.32260160011873423 }, plssvm::real_type{ 0.401642656885171 }, plssvm::real_type{ 0.05160647594201395 }, plssvm::real_type{ 1.224149267054074 }, plssvm::real_type{ -0.2415331131484474 }, plssvm::real_type{ -2.636779683484747e-16 }
    };

    if constexpr (classification == plssvm::classification_type::oaa) {
        // one vs. all classification
        ASSERT_EQ(rho.size(), num_classes_for_label_type);
        switch (num_classes_for_label_type) {
            case 2:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1] }));
                break;
            case 3:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
                break;
            case 4:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2], all_rhos[3] }));
                break;
            default:
                FAIL() << "Unreachable!";
        }
    } else if constexpr (classification == plssvm::classification_type::oao) {
        // one vs. all classification
        ASSERT_EQ(rho.size(), num_classes_for_label_type * (num_classes_for_label_type - 1) / 2);
        switch (num_classes_for_label_type) {
            case 2:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0] }));
                break;
            case 3:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, (std::vector<plssvm::real_type>{ all_rhos[0], all_rhos[1], all_rhos[2] }));
                break;
            case 4:
                EXPECT_FLOATING_POINT_VECTOR_EQ(rho, all_rhos);
                break;
            default:
                FAIL() << "Unreachable!";
        }
    } else {
        FAIL() << "unknown classification type";
    }

    // check labels
    EXPECT_EQ(label, util::get_correct_model_file_labels<label_type>());

    // check different classes
    EXPECT_EQ(different_classes.size(), num_classes_for_label_type);
    EXPECT_EQ(different_classes, util::get_distinct_label<label_type>());

    // check number of support vectors per class
    EXPECT_EQ(num_sv_per_class, util::get_correct_model_file_num_sv_per_class<label_type>(label.size()));

    // check number of header lines
    EXPECT_EQ(num_header_lines, this->correct_num_header_lines());
}
