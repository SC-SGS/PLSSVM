/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a performance tracker which can dump performance information in a YAML file.
 * @details Can be completely disabled during the CMake configuration step.
 */

#ifndef PLSSVM_DETAIL_PERFORMANCE_TRACKER_HPP_
#define PLSSVM_DETAIL_PERFORMANCE_TRACKER_HPP_
#pragma once

#include "plssvm/detail/assert.hpp"                      // PLSSVM_ASSERT
#include "plssvm/detail/cmd/parser_predict.hpp"          // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"            // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"            // plssvm::detail::cmd::parser_train
#include "plssvm/detail/type_traits.hpp"                 // plssvm::detail::remove_cvref_t
#include "plssvm/detail/utility.hpp"                     // plssvm::detail::current_date_time
#include "plssvm/version/git_metadata/git_metadata.hpp"  // plssvm::version::git_metadata::commit_sha1
#include "plssvm/version/version.hpp"                    // plssvm::version::{version, detail::target_platforms}

#include "fmt/chrono.h"   // format std::chrono types
#include "fmt/core.h"     // fmt::format
#include "fmt/ostream.h"  // format types with an operator<< overload

#include <fstream>        // std::ofstream
#include <iostream>       // std::ostream, std::ios_base::app
#include <string>         // std::string
#include <string_view>    // std::string_view
#include <type_traits>    // std::false_type, std::true_type
#include <unordered_map>  // std::unordered_map
#include <utility>        // std::move, std::pair

namespace plssvm::detail {

template <typename T>
struct tracking_entry {
    tracking_entry(const std::string_view category, const std::string_view name, const T value) :
        entry_category{ category }, entry_name{ name }, entry_value{ std::move(value) } {}

    const std::string entry_category{};
    const std::string_view entry_name{};
    const T entry_value{};
};

template <typename T>
std::ostream &operator<<(std::ostream &out, const tracking_entry<T> &entry) {
    return out << fmt::format("{}", entry.entry_value);
}

namespace impl {

template <typename T>
struct is_tracking_entry : std::false_type {};
template <typename T>
struct is_tracking_entry<tracking_entry<T>> : std::true_type {};

}  // namespace impl

template <typename T>
struct is_tracking_entry : impl::is_tracking_entry<detail::remove_cvref_t<T>> {};
template <typename T>
constexpr bool is_tracking_entry_v = is_tracking_entry<T>::value;

/**
 * @brief Store the tracked information during calls to `plssvm-train`, `plssvm-predict`, and `plssvm-scale`.
 * @details Can be completely disabled during the CMake configuration.
 */
class performance_tracker {
  public:
    /**
     * @brief Singleton getter since only one instance of a performance_tracker should exist.
     * @return the singleton instance (`[[nodiscard]]`)
     */
    [[nodiscard]] static performance_tracker &instance() {
        static performance_tracker tracker{};
        return tracker;
    }

    /**
     * @brief Add a tracking_entry to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     * @tparam T the type of the value the tracking_entry @p entry encapsulates
     * @param[in] entry the entry to add
     */
    template <typename T>
    void add_tracking_entry(const tracking_entry<T> &entry) {
        tracking_statistics.insert({ entry.entry_category, fmt::format("{}{}: {}\n", entry.entry_category.empty() ? "" : "  ", entry.entry_name, entry.entry_value) });
    }

    /**
     * @brief Add a tracking_entry encapsulating a std::string to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds quotes around the entry's value
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<std::string> &entry) {
        tracking_statistics.insert({ entry.entry_category, fmt::format("{}{}: \"{}\"\n", entry.entry_category.empty() ? "" : "  ", entry.entry_name, entry.entry_value) });
    }

    /**
     * @brief Add a tracking_entry encapsulating a plssvm::detail::cmd::parser_train to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the plssvm::detail::cmd::parser_train as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<cmd::parser_train> &entry) {
        tracking_statistics.insert({ entry.entry_category, fmt::format("  task:                        train\n"
                                                                       "  kernel_type:                 {}\n"
                                                                       "  degree:                      {}\n"
                                                                       "  gamma:                       {}\n"
                                                                       "  coef0:                       {}\n"
                                                                       "  cost:                        {}\n"
                                                                       "  epsilon:                     {}\n"
                                                                       "  max_iter:                    {}\n"
                                                                       "  backend:                     {}\n"
                                                                       "  target:                      {}\n"
                                                                       "  sycl_kernel_invocation_type: {}\n"
                                                                       "  sycl_implementation_type:    {}\n"
                                                                       "  strings_as_labels:           {}\n"
                                                                       "  float_as_real_type:          {}\n"
                                                                       "  input_filename:              \"{}\"\n"
                                                                       "  model_filename:              \"{}\"\n\n",
                                                                       entry.entry_value.csvm_params.kernel_type.value(),
                                                                       entry.entry_value.csvm_params.degree.value(),
                                                                       entry.entry_value.csvm_params.gamma.value(),
                                                                       entry.entry_value.csvm_params.coef0.value(),
                                                                       entry.entry_value.csvm_params.cost.value(),
                                                                       entry.entry_value.epsilon.value(),
                                                                       entry.entry_value.max_iter.value(),
                                                                       entry.entry_value.backend,
                                                                       entry.entry_value.target,
                                                                       entry.entry_value.sycl_kernel_invocation_type,
                                                                       entry.entry_value.sycl_implementation_type,
                                                                       entry.entry_value.strings_as_labels,
                                                                       entry.entry_value.float_as_real_type,
                                                                       entry.entry_value.input_filename,
                                                                       entry.entry_value.model_filename) });
    }
    /**
     * @brief Add a tracking_entry encapsulating a plssvm::detail::cmd::parser_predict to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the plssvm::detail::cmd::parser_predict as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<cmd::parser_predict> &entry) {
        tracking_statistics.insert({ entry.entry_category, fmt::format("  task:                     predict\n"
                                                                       "  backend:                  {}\n"
                                                                       "  target:                   {}\n"
                                                                       "  sycl_implementation_type: {}\n"
                                                                       "  strings_as_labels:        {}\n"
                                                                       "  float_as_real_type:       {}\n"
                                                                       "  input_filename:           \"{}\"\n"
                                                                       "  model_filename:           \"{}\"\n"
                                                                       "  predict_filename:         \"{}\"\n\n",
                                                                       entry.entry_value.backend,
                                                                       entry.entry_value.target,
                                                                       entry.entry_value.sycl_implementation_type,
                                                                       entry.entry_value.strings_as_labels,
                                                                       entry.entry_value.float_as_real_type,
                                                                       entry.entry_value.input_filename,
                                                                       entry.entry_value.model_filename,
                                                                       entry.entry_value.predict_filename) });
    }
    /**
     * @brief Add a tracking_entry encapsulating a plssvm::detail::cmd::parser_scale to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the plssvm::detail::cmd::parser_scale as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<cmd::parser_scale> &entry) {
        tracking_statistics.insert({ entry.entry_category, fmt::format("  task:               scale\n"
                                                                       "  lower:              {}\n"
                                                                       "  upper:              {}\n"
                                                                       "  format:             {}\n"
                                                                       "  strings_as_labels:  {}\n"
                                                                       "  float_as_real_type: {}\n"
                                                                       "  input_filename:     \"{}\"\n"
                                                                       "  scaled_filename:    \"{}\"\n"
                                                                       "  save_filename:      \"{}\"\n"
                                                                       "  restore_filename:   \"{}\"\n\n",
                                                                       entry.entry_value.lower,
                                                                       entry.entry_value.upper,
                                                                       entry.entry_value.format,
                                                                       entry.entry_value.strings_as_labels,
                                                                       entry.entry_value.float_as_real_type,
                                                                       entry.entry_value.input_filename,
                                                                       entry.entry_value.scaled_filename,
                                                                       entry.entry_value.save_filename,
                                                                       entry.entry_value.restore_filename) });
    }

    /**
     * @brief Write all stored tracking entries to the [YAML](https://yaml.org/) file @p filename.
     * @details Appends all entries at the end of the file creating a new YAML document.
     * @param[in] filename the file to add the performance tracking results to
     */
    void save(const std::string filename) {
        // append the current performance statistics to an already existing file if possible
        std::ofstream out{ filename, std::ios_base::app };
        PLSSVM_ASSERT(out.good(), fmt::format("Couldn't save performance tracking results in '{}'!", filename));

        // begin a new YAML document (only with "---" multiple YAML docments in a single file are allowed)
        out << "---\n";

        // output metadata information
        out << fmt::format(
            "meta_data:\n"
            "  date:                    \"{}\"\n"
            "  PLSSVM_TARGET_PLATFORMS: \"{}\"\n"
            "  commit:                  {}\n"
            "  version:                 {}\n"
            "\n",
            plssvm::detail::current_date_time(),
            version::detail::target_platforms,
            version::git_metadata::commit_sha1().empty() ? "unknown" : version::git_metadata::commit_sha1(),
            version::version);

        // output the actual (performance) statistics
        std::unordered_multimap<std::string, std::string>::iterator group_iter;  // iterate over all groups
        std::unordered_multimap<std::string, std::string>::iterator entry_iter;  // iterate over all entries in a specific group
        for (group_iter = tracking_statistics.begin(); group_iter != tracking_statistics.end(); group_iter = entry_iter) {
            // get the current group
            const std::string &group = group_iter->first;
            // find the range of all entries in the current group
            const std::pair key_range = tracking_statistics.equal_range(group);

            // output the group name, if it is not the empty string
            if (!group.empty()) {
                out << group << ":\n";
            }
            // output all performance statistic entries of the current group
            for (entry_iter = key_range.first; entry_iter != key_range.second; ++entry_iter) {
                out << fmt::format("{}", entry_iter->second);
            }
            out << '\n';
        }
    }

  private:
    /**
     * @brief Default construct a performance_tracker.
     * @details This constructor is private to enable a save singleton usage of the performance_tracker class.
     */
    performance_tracker() = default;

    /// All performance statistics grouped by their specified categories.
    std::unordered_multimap<std::string, std::string> tracking_statistics{};
};

/**
 * @def PLSSVM_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
 * @brief Defines the `PLSSVM_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Adds the provided entry to the plssvm::detail::performance_tracker singleton.
 */
/**
 * @def PLSSVM_PERFORMANCE_TRACKER_SAVE
 * @brief Defines the `PLSSVM_PERFORMANCE_TRACKER_SAVE` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Save the previously tracked performance statistics.
 */
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)

    #define PLSSVM_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(entry) \
        plssvm::detail::performance_tracker::instance().add_tracking_entry(entry)

    #define PLSSVM_PERFORMANCE_TRACKER_SAVE() \
        plssvm::detail::performance_tracker::instance().save(PLSSVM_PERFORMANCE_TRACKER_OUTPUT_FILE)

#else

    #define PLSSVM_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(entry)
    #define PLSSVM_PERFORMANCE_TRACKER_SAVE()

#endif

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_PERFORMANCE_TRACKER_HPP_
