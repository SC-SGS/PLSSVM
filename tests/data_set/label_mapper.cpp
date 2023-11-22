/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for functions related to the data_set label mapper used for learning an SVM model.
 */

#include "plssvm/data_set.hpp"

#include "plssvm/detail/string_conversion.hpp"  // plssvm::detail::convert_to
#include "plssvm/exceptions/exceptions.hpp"     // plssvm::data_set_exception

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT
#include "naming.hpp"              // naming::test_parameter_to_name
#include "types_to_test.hpp"       // util::{label_type_gtest, test_parameter_type_at_t}
#include "utility.hpp"             // util::{get_distinct_label, get_correct_data_file_labels}

#include "gtest/gtest.h"  // TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_DEATH, ASSERT_EQ, SUCCEED, ::testing::Test

#include <algorithm>    // std::find
#include <cstddef>      // std::size_t
#include <iterator>     // std::distance
#include <tuple>        // std::ignore
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

template <typename T>
class DataSetLabelMapper : public ::testing::Test {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
};
TYPED_TEST_SUITE(DataSetLabelMapper, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(DataSetLabelMapper, construct) {
    using label_type = typename TestFixture::fixture_label_type;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test values
    EXPECT_EQ(mapper.num_mappings(), distinct_labels.size());
    EXPECT_EQ(mapper.labels(), distinct_labels);
    // test mapping
    for (std::size_t i = 0; i < distinct_labels.size(); ++i) {
        EXPECT_EQ(mapper.get_label_by_mapped_index(i), distinct_labels[i]);
        EXPECT_EQ(mapper.get_mapped_index_by_label(distinct_labels[i]), i);
    }
}
TYPED_TEST(DataSetLabelMapper, get_mapped_index_by_label) {
    using label_type = typename TestFixture::fixture_label_type;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test the number of mappings
    ASSERT_EQ(mapper.num_mappings(), distinct_labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const std::size_t label_idx = std::distance(distinct_labels.cbegin(), std::find(distinct_labels.cbegin(), distinct_labels.cend(), labels[i]));
        EXPECT_EQ(mapper.get_mapped_index_by_label(labels[i]), label_idx);
    }
}
TYPED_TEST(DataSetLabelMapper, get_mapped_index_by_invalid_label) {
    using label_type = typename TestFixture::fixture_label_type;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test the number of mappings
    if constexpr (!std::is_same_v<label_type, bool>) {
        // can't have an unknown labels for bool
        EXPECT_THROW_WHAT(std::ignore = mapper.get_mapped_index_by_label(plssvm::detail::convert_to<label_type>("9")),
                          plssvm::data_set_exception,
                          R"(Label "9" unknown in this label mapping!)");
    } else {
        SUCCEED() << "By definition there can't be unknown labels for the boolean label type.";
    }
}
TYPED_TEST(DataSetLabelMapper, get_label_by_mapped_index) {
    using label_type = typename TestFixture::fixture_label_type;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test the number of mappings
    ASSERT_EQ(mapper.num_mappings(), distinct_labels.size());
    for (std::size_t i = 0; i < labels.size(); ++i) {
        const std::size_t label_idx = std::distance(distinct_labels.cbegin(), std::find(distinct_labels.cbegin(), distinct_labels.cend(), labels[i]));
        EXPECT_EQ(mapper.get_label_by_mapped_index(label_idx), labels[i]);
    }
}
TYPED_TEST(DataSetLabelMapper, get_label_by_invalid_mapped_index) {
    using label_type = typename TestFixture::fixture_label_type;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> distinct_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ distinct_labels };

    // test the number of mappings
    EXPECT_THROW_WHAT(std::ignore = mapper.get_label_by_mapped_index(mapper.num_mappings() + 1),
                      plssvm::data_set_exception,
                      fmt::format("Mapped index \"{}\" unknown in this label mapping!", mapper.num_mappings() + 1));
}
TYPED_TEST(DataSetLabelMapper, num_mappings) {
    using label_type = typename TestFixture::fixture_label_type;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the number of mappings
    EXPECT_EQ(mapper.num_mappings(), different_labels.size());
}
TYPED_TEST(DataSetLabelMapper, labels) {
    using label_type = typename TestFixture::fixture_label_type;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // the different labels
    const std::vector<label_type> different_labels = util::get_distinct_label<label_type>();

    // create label mapper
    const label_mapper_type mapper{ different_labels };

    // test the different labels
    EXPECT_EQ(mapper.labels(), different_labels);
}

template <typename T>
class DataSetLabelMapperDeathTest : public DataSetLabelMapper<T> {};
TYPED_TEST_SUITE(DataSetLabelMapperDeathTest, util::label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(DataSetLabelMapperDeathTest, duplicated_labels) {
    using label_type = typename TestFixture::fixture_label_type;
    using label_mapper_type = typename plssvm::data_set<label_type>::label_mapper;

    // duplicated labels are not allowed in the label_mapper constructor
    EXPECT_DEATH(label_mapper_type{ util::get_correct_data_file_labels<label_type>() },
                 "The provided labels for the label_mapper must not include duplicated ones!");
}