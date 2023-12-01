/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for writing a LIBSVM model file header.
 */

#include "plssvm/detail/io/libsvm_model_parsing.hpp"

#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "naming.hpp"         // naming::label_type_to_name
#include "types_to_test.hpp"  // util::label_type_gtest
#include "utility.hpp"        // util::{temporary_file, get_distinct_label, get_correct_model_file_labels, get_correct_model_file_num_sv_per_class,
                              // generate_specific_matrix, get_num_classes}

#include "fmt/format.h"            // fmt::format, fmt::join
#include "fmt/os.h"                // fmt::ostream, fmt::output_file
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_DEATH, ASSERT_EQ, ::testing::Test

#include <cstddef>  // std::size_t
#include <tuple>    // std::ignore
#include <vector>   // std::vector

template <typename T>
class LIBSVMModelHeaderWrite : public ::testing::Test, protected util::temporary_file {};
TYPED_TEST_SUITE(LIBSVMModelHeaderWrite, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelHeaderWrite, write_linear) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const std::size_t num_rho_values = distinct_label.size() * (distinct_label.size() - 1) / 2;  // OAO -> tests don't change for OAA
    const std::vector<label_type> label = util::get_correct_model_file_labels<label_type>();
    const std::vector<std::size_t> num_sv = util::get_correct_model_file_num_sv_per_class<label_type>(label.size());
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label.size(), 3);

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::linear };
    const std::vector<plssvm::real_type> rho(num_rho_values, plssvm::real_type{ 3.14159265359 });
    const plssvm::data_set<label_type> data_set{ data, std::vector<label_type>{ label } };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    const std::vector<label_type> &label_order = plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set);
    out.close();

    // check returned label order
    EXPECT_EQ(label_order, distinct_label);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 8);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type linear");
    EXPECT_EQ(reader.line(2), fmt::format("nr_class {}", distinct_label.size()));
    EXPECT_EQ(reader.line(3), fmt::format("label {}", fmt::join(distinct_label, " ")));
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(5), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(6), fmt::format("rho {:.10e}", fmt::join(rho, " ")));
    EXPECT_EQ(reader.line(7), "SV");
}
TYPED_TEST(LIBSVMModelHeaderWrite, write_polynomial) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const std::size_t num_rho_values = distinct_label.size() * (distinct_label.size() - 1) / 2;  // OAO -> tests don't change for OAA
    const std::vector<label_type> label = util::get_correct_model_file_labels<label_type>();
    const std::vector<std::size_t> num_sv = util::get_correct_model_file_num_sv_per_class<label_type>(label.size());
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label.size(), 3);

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial, plssvm::degree = 3, plssvm::gamma = 2.2, plssvm::coef0 = 4.4 };
    const std::vector<plssvm::real_type> rho(num_rho_values, plssvm::real_type{ 3.14159265359 });
    const plssvm::data_set<label_type> data_set{ data, std::vector<label_type>{ label } };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    const std::vector<label_type> &label_order = plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set);
    out.close();

    // check returned label order
    EXPECT_EQ(label_order, distinct_label);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 11);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type polynomial");
    EXPECT_EQ(reader.line(2), "degree 3");
    EXPECT_EQ(reader.line(3), "gamma 2.2");
    EXPECT_EQ(reader.line(4), "coef0 4.4");
    EXPECT_EQ(reader.line(5), fmt::format("nr_class {}", distinct_label.size()));
    EXPECT_EQ(reader.line(6), fmt::format("label {}", fmt::join(distinct_label, " ")));
    EXPECT_EQ(reader.line(7), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(8), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(9), fmt::format("rho {:.10e}", fmt::join(rho, " ")));
    EXPECT_EQ(reader.line(10), "SV");
}
TYPED_TEST(LIBSVMModelHeaderWrite, write_rbf) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // define data to write
    const std::vector<label_type> distinct_label = util::get_distinct_label<label_type>();
    const std::size_t num_rho_values = distinct_label.size() * (distinct_label.size() - 1) / 2;  // OAO -> tests don't change for OAA
    const std::vector<label_type> label = util::get_correct_model_file_labels<label_type>();
    const std::vector<std::size_t> num_sv = util::get_correct_model_file_num_sv_per_class<label_type>(label.size());
    const auto data = util::generate_specific_matrix<plssvm::aos_matrix<plssvm::real_type>>(label.size(), 3);

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.4 };
    const std::vector<plssvm::real_type> rho(num_rho_values, plssvm::real_type{ 3.14159265359 });
    const plssvm::data_set<label_type> data_set{ data, std::vector<label_type>{ label } };

    // write the LIBSVM model to the temporary file
    fmt::ostream out = fmt::output_file(this->filename);
    const std::vector<label_type> &label_order = plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set);
    out.close();

    // check returned label order
    EXPECT_EQ(label_order, distinct_label);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 9);  // the LIBSVM header
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type rbf");
    EXPECT_EQ(reader.line(2), "gamma 0.4");
    EXPECT_EQ(reader.line(3), fmt::format("nr_class {}", distinct_label.size()));
    EXPECT_EQ(reader.line(4), fmt::format("label {}", fmt::join(distinct_label, " ")));
    EXPECT_EQ(reader.line(5), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(6), fmt::format("nr_sv {}", fmt::join(num_sv, " ")));
    EXPECT_EQ(reader.line(7), fmt::format("rho {:.10e}", fmt::join(rho, " ")));
    EXPECT_EQ(reader.line(8), "SV");
}

template <typename T>
class LIBSVMModelHeaderWriteDeathTest : public LIBSVMModelHeaderWrite<T>, private util::redirect_output<> {};
TYPED_TEST_SUITE(LIBSVMModelHeaderWriteDeathTest, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelHeaderWriteDeathTest, write_header_without_label) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::size_t num_classes = util::get_num_classes<label_type>();
    const std::vector<plssvm::real_type> rho(num_classes * (num_classes - 1) / 2);  // OAO -> tests don't change for OAA
    const plssvm::data_set<label_type> data_set{ std::vector<std::vector<plssvm::real_type>>{ { plssvm::real_type{ 0.0 } } } };

    // create file
    fmt::ostream out = fmt::output_file(this->filename);

    // try writing the LIBSVM model header
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set)),
                 "Cannot write a model file that does not include labels!");
}
TYPED_TEST(LIBSVMModelHeaderWriteDeathTest, write_header_invalid_number_of_rho_values) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;

    // create necessary parameter
    const plssvm::parameter params{};
    const std::vector<plssvm::real_type> rho{};
    const plssvm::data_set<label_type> data_set{ std::vector<std::vector<plssvm::real_type>>{ { plssvm::real_type{ 0.0 } } },
                                                 std::vector<label_type>{ util::get_distinct_label<label_type>().front() } };

    // create file
    fmt::ostream out = fmt::output_file(this->filename);

    // try writing the LIBSVM model header
    EXPECT_DEATH(std::ignore = (plssvm::detail::io::write_libsvm_model_header(out, params, rho, data_set)),
                 ::testing::HasSubstr("At least one rho value must be provided!"));
}