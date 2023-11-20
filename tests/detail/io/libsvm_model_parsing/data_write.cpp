/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for writing LIBSVM model file data section.
 */

#include "plssvm/detail/io/libsvm_model_parsing.hpp"

#include "plssvm/classification_types.hpp"   // plssvm::classification_type, plssvm::calculate_number_of_classifiers
#include "plssvm/constants.hpp"              // plssvm::real_type
#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/detail/io/file_reader.hpp"  // plssvm::detail::io::file_reader
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "naming.hpp"         // naming::parameter_definition_to_name
#include "types_to_test.hpp"  // util::label_type_classification_type_gtest
#include "utility.hpp"        // util::{get_distinct_label, get_correct_model_file_labels, get_correct_model_file_num_sv_per_class,
                              // generate_random_matrix, get_num_classes, generate_random_vector}

#include "fmt/format.h"            // fmt::format, fmt::join
#include "gmock/gmock-matchers.h"  // ::testing::HasSubstr
#include "gtest/gtest.h"           // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_DEATH, ASSERT_EQ, FAIL, SUCCEED, ::testing::Test

#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <vector>       // std::vector

template <typename T>
class LIBSVMModelDataWrite : public ::testing::Test, private util::redirect_output<>, protected util::temporary_file {};
TYPED_TEST_SUITE(LIBSVMModelDataWrite, util::label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelDataWrite, write) {
    using label_type = util::test_parameter_type_at_t<0, TypeParam>;
    constexpr plssvm::classification_type classification = util::test_parameter_value_at_v<0, TypeParam>;

    // define data to write
    const std::vector<label_type> classes = util::get_distinct_label<label_type>();
    const std::size_t num_classes = classes.size();
    const std::size_t num_classifiers = plssvm::calculate_number_of_classifiers(classification, classes.size());
    const std::vector<label_type> label = util::get_correct_model_file_labels<label_type>();
    const std::vector<std::size_t> num_sv_per_class = util::get_correct_model_file_num_sv_per_class<label_type>(label.size());
    const auto data = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(label.size(), 3);

    // create necessary parameter
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::linear };
    const std::vector<plssvm::real_type> rho(num_classifiers, plssvm::real_type{ 3.1415 });
    std::vector<std::vector<std::size_t>> index_sets{};
    if constexpr (classification == plssvm::classification_type::oao) {
        switch (num_classes) {
            case 2:
                // [ 1, 1, 1, 0, 0, 0 ]
                index_sets.push_back(std::vector<std::size_t>{ 0, 1, 2 });
                index_sets.push_back(std::vector<std::size_t>{ 3, 4, 5 });
                break;
            case 3:
                // [ 1, 1, 2, 2, 3, 3 ]
                index_sets.push_back(std::vector<std::size_t>{ 0, 1 });
                index_sets.push_back(std::vector<std::size_t>{ 2, 3 });
                index_sets.push_back(std::vector<std::size_t>{ 4, 5 });
                break;
            case 4:
                // [ 1, 1, 2, 2, 3, 4 ]
                index_sets.push_back(std::vector<std::size_t>{ 0, 1 });
                index_sets.push_back(std::vector<std::size_t>{ 2, 3 });
                index_sets.push_back(std::vector<std::size_t>{ 4 });
                index_sets.push_back(std::vector<std::size_t>{ 5 });
                break;
            default:
                FAIL() << "Invalid number of classes!";
        }
    }
    std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha{};
    if constexpr (classification == plssvm::classification_type::oaa) {
        alpha.emplace_back(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(num_classifiers, data.num_rows()));
    } else if constexpr (classification == plssvm::classification_type::oao) {
        for (std::size_t i = 0; i < num_classes; ++i) {
            for (std::size_t j = i + 1; j < num_classes; ++j) {
                alpha.emplace_back(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(1, index_sets[i].size() + index_sets[j].size()));
            }
        }
    } else {
        FAIL() << "Unknown classification type!";
    }
    const plssvm::data_set<label_type> data_set{ data, std::vector<label_type>{ label } };

    // write the LIBSVM model file
    plssvm::detail::io::write_libsvm_model_data(this->filename, params, classification, rho, alpha, index_sets, data_set);

    // read the written file
    plssvm::detail::io::file_reader reader{ this->filename };
    reader.read_lines('#');

    // check the written data
    ASSERT_EQ(reader.num_lines(), 8 + data.num_rows());  // header lines + data
    EXPECT_EQ(reader.line(0), "svm_type c_svc");
    EXPECT_EQ(reader.line(1), "kernel_type linear");
    EXPECT_EQ(reader.line(2), fmt::format("nr_class {}", num_classes));
    EXPECT_EQ(reader.line(3), fmt::format("label {}", fmt::join(classes, " ")));
    EXPECT_EQ(reader.line(4), fmt::format("total_sv {}", data.num_rows()));
    EXPECT_EQ(reader.line(5), fmt::format("nr_sv {}", fmt::join(num_sv_per_class, " ")));
    EXPECT_EQ(reader.line(6), fmt::format("rho {:.10e}", fmt::join(rho, " ")));
    EXPECT_EQ(reader.line(7), "SV");

    // iterate over all classes
    std::size_t idx{ 0 };
    for (std::size_t i = 0; i < num_classes; ++i) {
        // iterate over all support vectors for one class IN THE FILE
        for (std::size_t j = 0; j < num_sv_per_class[i]; ++j) {
            // check whether the necessary rows for the support vectors of the current class are present
            const std::string_view read_line = reader.line(8 + idx + j);
            int line_found{ 0 };

            std::string correct_line{};
            if constexpr (classification == plssvm::classification_type::oaa) {
                for (std::size_t k = 0; k < num_sv_per_class[i]; ++k) {
                    // iterate over all support vectors for one class IN THE ORIGINAL DATA
                    // assemble the alpha vector
                    std::vector<plssvm::real_type> alpha_vec{};
                    for (std::size_t l = 0; l < num_classifiers; ++l) {
                        alpha_vec.push_back(alpha.front()(l, idx + k));
                    }
                    correct_line = fmt::format("{:.10e} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", fmt::join(alpha_vec, " "), data(idx + k, 0), data(idx + k, 1), data(idx + k, 2));

                    if (read_line == correct_line) {
                        ++line_found;
                    }
                }
            } else if constexpr (classification == plssvm::classification_type::oao) {
                for (std::size_t k = 0; k < num_sv_per_class[i]; ++k) {
                    const std::size_t sv_idx = idx + k;
                    // iterate over all support vectors for one class IN THE ORIGINAL DATA
                    // assemble the alpha vector
                    std::vector<plssvm::real_type> alpha_vec{};
                    switch (num_classes) {
                        case 2:
                            alpha_vec.emplace_back(alpha.front()(0, sv_idx));
                            break;
                        case 3: {
                            ASSERT_EQ(alpha.size(), 3);
                            static const std::vector<std::vector<plssvm::real_type>> correct_output_alpha_vec_three_classes{
                                // alpha order: 0v1 -> 0v2 -> 1v2
                                // idx:          0  ->  1  ->  2
                                // size:         4  ->  4  ->  4
                                { alpha[0](0, 0), alpha[1](0, 0) },  // data point 0 ->       0v1 |       0v2
                                { alpha[0](0, 1), alpha[1](0, 1) },  // data point 1 ->       0v1 |       0v2
                                { alpha[0](0, 2), alpha[2](0, 0) },  // data point 2 -> 1v0 = 0v1 |       1v2
                                { alpha[0](0, 3), alpha[2](0, 1) },  // data point 3 -> 1v0 = 0v1 |       1v2
                                { alpha[1](0, 2), alpha[2](0, 2) },  // data point 4 -> 2v0 = 0v2 | 2v1 = 1v2
                                { alpha[1](0, 3), alpha[2](0, 3) },  // data point 5 -> 2v0 = 0v2 | 2v1 = 1v2
                            };
                            alpha_vec = correct_output_alpha_vec_three_classes[sv_idx];
                        } break;
                        case 4: {
                            ASSERT_EQ(alpha.size(), 6);
                            static const std::vector<std::vector<plssvm::real_type>> correct_output_alpha_vec_four_classes{
                                // alpha order: 0v1 -> 0v2 -> 0v3 -> 1v2 -> 1v3 -> 2v3
                                // idx:          0  ->  1  ->  2  ->  3  ->  4  ->  5
                                // size:         4  ->  3  ->  3  ->  3  ->  3  ->  2
                                { alpha[0](0, 0), alpha[1](0, 0), alpha[2](0, 0) },  // data point 0 ->       0v1 |       0v2 |       0v3
                                { alpha[0](0, 1), alpha[1](0, 1), alpha[2](0, 1) },  // data point 1 ->       0v1 |       0v2 |       0v3
                                { alpha[0](0, 2), alpha[3](0, 0), alpha[4](0, 0) },  // data point 2 -> 1v0 = 0v1 |       1v2 |       1v3
                                { alpha[0](0, 3), alpha[3](0, 1), alpha[4](0, 1) },  // data point 3 -> 1v0 = 0v1 |       1v2 |       1v3
                                { alpha[1](0, 2), alpha[3](0, 2), alpha[5](0, 0) },  // data point 4 -> 2v0 = 0v2 | 2v1 = 1v2 |       2v3
                                { alpha[2](0, 2), alpha[4](0, 2), alpha[5](0, 1) },  // data point 5 -> 3v0 = 0v3 | 3v1 = 1v3 | 3v2 = 2v3
                            };
                            alpha_vec = correct_output_alpha_vec_four_classes[sv_idx];
                        } break;
                        default:
                            FAIL() << "Invalid number of classes!";
                    }
                    correct_line = fmt::format("{:.10e} 1:{:.10e} 2:{:.10e} 3:{:.10e} ", fmt::join(alpha_vec, " "), data(sv_idx, 0), data(sv_idx, 1), data(sv_idx, 2));

                    if (read_line == correct_line) {
                        ++line_found;
                    }
                }
            } else {
                FAIL() << "Unknown classification type!";
            }

            // check, how often the line in the file was found in the original data
            if (line_found == 0) {
                FAIL() << fmt::format("Couldn't find the line '{}' ({}) from the output file in the provided data set.", read_line, idx);
            } else if (line_found > 1) {
                FAIL() << fmt::format("Could find the line '{}' ({}) from the output file in the provided data set multiple times.", read_line, idx);
            }
        }
        idx += num_sv_per_class[i];
    }
    SUCCEED();
}

template <typename T>
class LIBSVMModelDataWriteDeathTest : public LIBSVMModelDataWrite<T> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    static constexpr plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<0, T>;

    void SetUp() override {
        const std::size_t num_classes = util::get_num_classes<fixture_label_type>();
        rho_ = util::generate_random_vector<plssvm::real_type>(this->num_classifiers());
        // create the weight vector based on the classification type and number of classes
        switch (fixture_classification) {
            case plssvm::classification_type::oaa:
                alpha_.emplace_back(num_classes, 6);
                break;
            case plssvm::classification_type::oao:
                switch (num_classes) {
                    case 2:
                        alpha_.emplace_back(1, 6);
                        break;
                    case 3:
                        alpha_.emplace_back(1, 4);
                        alpha_.emplace_back(1, 4);
                        alpha_.emplace_back(1, 4);
                        break;
                    case 4:
                        alpha_.emplace_back(1, 4);
                        alpha_.emplace_back(1, 3);
                        alpha_.emplace_back(1, 3);
                        alpha_.emplace_back(1, 3);
                        alpha_.emplace_back(1, 3);
                        alpha_.emplace_back(1, 2);
                        break;
                    default:
                        FAIL();
                }
                break;
        }
        // create index sets based on the number of classes
        switch (num_classes) {
            case 2:
                index_sets_ = std::vector<std::vector<std::size_t>>{ { 0, 1, 2 }, { 3, 4, 5 } };
                break;
            case 3:
                index_sets_ = std::vector<std::vector<std::size_t>>{ { 0, 1 }, { 2, 3 }, { 4, 5 } };
                break;
            case 4:
                index_sets_ = std::vector<std::vector<std::size_t>>{ { 0, 1 }, { 2, 3 }, { 4 }, { 5 } };
                break;
            default:
                FAIL();
        }
    }

    /**
     * @brief Return the default parameter.
     * @return the parameter (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::parameter &get_params() const noexcept { return params_; }
    /**
     * @brief Return the rho values.
     * @details The size depends on the used classification type and number of classes.
     * @return the rho values (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<plssvm::real_type> &get_rho() const noexcept { return rho_; }
    /**
     * @brief Return the weights.
     * @details The shape of the vector and the containing matrices depend on the used classification type and number of classes.
     * @return the weights (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<plssvm::aos_matrix<plssvm::real_type>> &get_alpha() const noexcept { return alpha_; }
    /**
     * @brief Return the index sets indicating which data point is a support vector for which class.
     * @return the index sets (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<std::vector<std::size_t>> &get_index_sets() const noexcept { return index_sets_; }
    /**
     * @brief Return the data set containing all support vectors.
     * @return the support vectors (`[[nodiscard]]`)
     */
    [[nodiscard]] const plssvm::data_set<fixture_label_type> &get_data_set() const noexcept { return data_set_; }

    /**
     * @brief Return the number of classifiers depending on the used classification type and number of classes.
     * @return the number of classifiers (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t num_classifiers() const noexcept { return plssvm::calculate_number_of_classifiers(fixture_classification, util::get_num_classes<fixture_label_type>()); }

  private:
    /// The default parameters.
    plssvm::parameter params_{};
    /// The rho vector; size depending on used classification type and number of classes.
    std::vector<plssvm::real_type> rho_{};
    /// The weights; shape of the vector and the containing matrices depending on used classification type and number of classes.
    std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha_{};
    /// The index sets indicating which data point is a support vector for which class.
    std::vector<std::vector<std::size_t>> index_sets_{};
    /// The support vectors.
    plssvm::data_set<fixture_label_type> data_set_{ util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(6, 2), util::get_correct_model_file_labels<fixture_label_type>() };
};
TYPED_TEST_SUITE(LIBSVMModelDataWriteDeathTest, util::label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(LIBSVMModelDataWriteDeathTest, empty_filename) {
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data("", this->get_params(), classification, this->get_rho(), this->get_alpha(), this->get_index_sets(), this->get_data_set())),
                 "The provided model filename must not be empty!");
}
TYPED_TEST(LIBSVMModelDataWriteDeathTest, missing_labels) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create invalid parameter
    const plssvm::data_set<label_type> data_set{ util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(4, 2) };

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), this->get_index_sets(), data_set)),
                 "Cannot write a model file that does not include labels!");
}
TYPED_TEST(LIBSVMModelDataWriteDeathTest, invalid_number_of_rho_values) {
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create invalid parameter
    const std::vector<plssvm::real_type> rho = util::generate_random_vector<plssvm::real_type>(42);

    // try writing the LIBSVM model header
    EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, rho, this->get_alpha(), this->get_index_sets(), this->get_data_set())),
                 ::testing::HasSubstr(fmt::format("The number of rho values is 42 but must be {} ({})", this->num_classifiers(), classification)));
}
TYPED_TEST(LIBSVMModelDataWriteDeathTest, invalid_alpha_vector) {
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    if constexpr (classification == plssvm::classification_type::oaa) {
        {
            // alpha vector too large
            const std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha(2);
            EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), alpha, this->get_index_sets(), this->get_data_set())),
                         "In case of OAA, the alpha vector may only contain one matrix as entry, but has 2!");
        }
        {
            // invalid number of rows in matrix
            const std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha{ plssvm::aos_matrix<plssvm::real_type>{ 42, 6 } };
            EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), alpha, this->get_index_sets(), this->get_data_set())),
                         fmt::format("The number of rows in the matrix must be {}, but is 42!", this->num_classifiers(), classification));
        }
        {
            // invalid number of columns in matrix
            const std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha{ plssvm::aos_matrix<plssvm::real_type>{ this->num_classifiers(), 42 } };
            EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), alpha, this->get_index_sets(), this->get_data_set())),
                         ::testing::HasSubstr("The number of weights (42) must be equal to the number of support vectors (6)!"));
        }
    } else if constexpr (classification == plssvm::classification_type::oao) {
        {
            // alpha vector too large
            const std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha(42);
            EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), alpha, this->get_index_sets(), this->get_data_set())),
                         fmt::format("The number of matrices in the alpha vector must contain {} entries, but contains 42 entries!", this->num_classifiers()));
        }
        {
            // invalid matrix shape
            std::vector<plssvm::aos_matrix<plssvm::real_type>> alpha(this->get_alpha());
            alpha.back() = plssvm::aos_matrix<plssvm::real_type>{ 3, 2 };
            EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), alpha, this->get_index_sets(), this->get_data_set())),
                         "In case of OAO, each matrix may only contain one row!");
        }
    } else {
        FAIL() << "Invalid classification_type!";
    }
}
TYPED_TEST(LIBSVMModelDataWriteDeathTest, invalid_number_of_index_sets) {
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create invalid parameter
    std::vector<std::vector<std::size_t>> index_sets = this->get_index_sets();
    index_sets.pop_back();

    // try writing the LIBSVM model header
    if constexpr (classification == plssvm::classification_type::oaa) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     fmt::format("There shouldn't be any index sets for the OAA classification, but {} were found!", this->get_index_sets().size() - 1));
    } else if constexpr (classification == plssvm::classification_type::oao) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     ::testing::HasSubstr(fmt::format("The number of index sets ({}) must be equal to the number of different classes ({})!", index_sets.size(), this->get_index_sets().size())));
    } else {
        FAIL() << "Invalid classification_type!";
    }
}
TYPED_TEST(LIBSVMModelDataWriteDeathTest, invalid_number_of_indices) {
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create invalid parameter
    std::vector<std::vector<std::size_t>> index_sets = this->get_index_sets();
    index_sets.front().pop_back();

    // try writing the LIBSVM model header
    if constexpr (classification == plssvm::classification_type::oaa) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     fmt::format("There shouldn't be any index sets for the OAA classification, but {} were found!", this->get_index_sets().size()));
    } else if constexpr (classification == plssvm::classification_type::oao) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     "Each data point must have exactly one entry in the index set!");
    } else {
        FAIL() << "Invalid classification_type!";
    }
}
TYPED_TEST(LIBSVMModelDataWriteDeathTest, indices_not_sorted) {
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create invalid parameter
    std::vector<std::vector<std::size_t>> index_sets = this->get_index_sets();
    index_sets.front().front() = 42;

    // try writing the LIBSVM model header
    if constexpr (classification == plssvm::classification_type::oaa) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     fmt::format("There shouldn't be any index sets for the OAA classification, but {} were found!", this->get_index_sets().size()));
    } else if constexpr (classification == plssvm::classification_type::oao) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     "All index sets must be sorted in ascending order!");
    } else {
        FAIL() << "Invalid classification_type!";
    }
}
TYPED_TEST(LIBSVMModelDataWriteDeathTest, indices_in_one_index_set_not_unique) {
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create invalid parameter
    std::vector<std::vector<std::size_t>> index_sets = this->get_index_sets();
    index_sets.front().front() = 1;

    // try writing the LIBSVM model header
    if constexpr (classification == plssvm::classification_type::oaa) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     fmt::format("There shouldn't be any index sets for the OAA classification, but {} were found!", this->get_index_sets().size()));
    } else if constexpr (classification == plssvm::classification_type::oao) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     "All indices in one index set must be unique!");
    } else {
        FAIL() << "Invalid classification_type!";
    }
}
TYPED_TEST(LIBSVMModelDataWriteDeathTest, index_sets_not_disjoint) {
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create invalid parameter
    std::vector<std::vector<std::size_t>> index_sets = this->get_index_sets();
    index_sets.front().back() = 4;

    // try writing the LIBSVM model header
    if constexpr (classification == plssvm::classification_type::oaa) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     fmt::format("There shouldn't be any index sets for the OAA classification, but {} were found!", this->get_index_sets().size()));
    } else if constexpr (classification == plssvm::classification_type::oao) {
        EXPECT_DEATH((plssvm::detail::io::write_libsvm_model_data(this->filename, this->get_params(), classification, this->get_rho(), this->get_alpha(), index_sets, this->get_data_set())),
                     fmt::format("All index sets must be pairwise unique, but index sets 0 and {} share at least one index!", this->get_data_set().num_classes() == 2 ? 1 : 2));
    } else {
        FAIL() << "Invalid classification_type!";
    }
}
