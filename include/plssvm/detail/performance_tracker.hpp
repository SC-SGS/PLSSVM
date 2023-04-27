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

#include "plssvm/detail/cmd/parser_predict.hpp"  // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"    // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"    // plssvm::detail::cmd::parser_train
#include "plssvm/detail/type_traits.hpp"         // plssvm::detail::remove_cvref_t

#include "fmt/chrono.h"                          // format std::chrono types
#include "fmt/core.h"                            // fmt::format
#include "fmt/ostream.h"                         // format types with an operator<< overload

#include <string>                                // std::string
#include <string_view>                           // std::string_view
#include <type_traits>                           // std::false_type, std::true_type
#include <unordered_map>                         // std::unordered_multimap
#include <utility>                               // std::move

namespace plssvm::detail {

/**
 * @brief A single tracking entry containing a specific category, a unique name, and the actual value to be tracked.
 * @tparam T the type of the value to be tracked
 */
template <typename T>
struct tracking_entry {
    /**
     * @brief Create a new tracking entry.
     * @param[in] category the category to which this tracking entry belongs; used for grouping in the resulting YAML file
     * @param[in] name the name of the tracking entry
     * @param[in] value the tracked value
     */
    tracking_entry(const std::string_view category, const std::string_view name, T value) :
        entry_category{ category }, entry_name{ name }, entry_value{ std::move(value) } {}

    /// The category to which this tracking entry belongs; used for grouping in the resulting YAML file.
    const std::string entry_category{};
    /// The name of the tracking entry displayed in the YAML file.
    const std::string_view entry_name{};
    /// The tracked value in the YAML file.
    const T entry_value{};
};

/**
 * @brief Output the tracking @p entry to the given output-stream @p out. Only the tracked value is output **excluding** the category and name.
 * @tparam T the type of the value tracked in the tracking @p entry
 * @param[in,out] out the output-stream to write the tracking entry to
 * @param[in] entry the tracking entry
 * @return the output-stream
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, const tracking_entry<T> &entry) {
    return out << fmt::format("{}", entry.entry_value);
}

namespace impl {

/**
 * @brief Sets the `value` to `false` since it **isn't** a tracking entry.
 */
template <typename T>
struct is_tracking_entry : std::false_type {};
/**
 * @brief Sets the `value` to `true` since it **is** a tracking entry.
 */
template <typename T>
struct is_tracking_entry<tracking_entry<T>> : std::true_type {};

}  // namespace impl

/**
 * @brief Check whether @p T is a tracking entry. Ignores all top-level const, volatile, and reference qualifiers.
 * @tparam T the type to check whether it is a tracking entry or not
 */
template <typename T>
struct is_tracking_entry : impl::is_tracking_entry<detail::remove_cvref_t<T>> {};
/**
 * @copydoc plssvm::is_tracking_entry
 * @details A shorthand for `plssvm::is_tracking_entry::value`.
 */
template <typename T>
constexpr bool is_tracking_entry_v = is_tracking_entry<T>::value;

/**
 * @brief Store the tracked information during calls to `plssvm-train`, `plssvm-predict`, and `plssvm-scale`.
 * @details Can be completely disabled during the CMake configuration.
 */
class performance_tracker {
  public:
    /**
     * @brief Singleton getter since only one instance of a performance_tracker should ever exist.
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
        tracking_statistics.emplace(entry.entry_category, fmt::format("{}{}: {}\n", entry.entry_category.empty() ? "" : "  ", entry.entry_name, entry.entry_value));
    }

    /**
     * @brief Add a tracking_entry encapsulating a std::string to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds quotes around the entry's value.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<std::string> &entry);
    /**
     * @brief Add a tracking_entry encapsulating a plssvm::detail::cmd::parser_train to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the plssvm::detail::cmd::parser_train as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<cmd::parser_train> &entry);
    /**
     * @brief Add a tracking_entry encapsulating a plssvm::detail::cmd::parser_predict to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the plssvm::detail::cmd::parser_predict as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<cmd::parser_predict> &entry);
    /**
     * @brief Add a tracking_entry encapsulating a plssvm::detail::cmd::parser_scale to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the plssvm::detail::cmd::parser_scale as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<cmd::parser_scale> &entry);

    /**
     * @brief Write all stored tracking entries to the [YAML](https://yaml.org/) file @p filename.
     * @details Appends all entries at the end of the file creating a new YAML document.
     * @param[in] filename the file to add the performance tracking results to
     */
    void save(const std::string &filename);

    /**
     * @brief Pause the current tracking, i.e., all calls to `add_tracking_entry` do nothing.
     */
    void pause_tracking() noexcept { is_tracking_ = false; }
    /**
     * @brief Resume the tracking, i.e., all calls to `add_tracking_entry` do track the values again.
     */
    void resume_tracking() noexcept { is_tracking_ = true; }
    /**
     * @brief Check whether tracking is currently active or paused.
     * @return `true` if tracking is enabled, `false` if it is currently paused (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_tracking() const noexcept { return is_tracking_; }

    /**
     * @brief Return the currently available tracking entries.
     * @return the previously added tracking entries (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::unordered_multimap<std::string, std::string> &get_tracking_entries() const noexcept { return tracking_statistics; }

  private:
    /**
     * @brief Default construct a performance_tracker.
     * @details This constructor is private to enable a save singleton usage of the performance_tracker class.
     */
    performance_tracker() = default;

    /// All performance statistics grouped by their specified categories.
    std::unordered_multimap<std::string, std::string> tracking_statistics{};
    /// The tracking is enabled by default.
    bool is_tracking_{ true };
};

/**
 * @def PLSSVM_DETAIL_PERFORMANCE_TRACKER_PAUSE
 * @brief Defines the `PLSSVM_DETAIL_PERFORMANCE_TRACKER_PAUSE` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Pause the tracking functionality if tracking is currently enabled.
 */
/**
 * @def PLSSVM_DETAIL_PERFORMANCE_TRACKER_RESUME
 * @brief Defines the `PLSSVM_DETAIL_PERFORMANCE_TRACKER_RESUME` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Resume the tracking functionality if tracking is currently enabled.
 */
/**
 * @def PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE
 * @brief Defines the `PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Save the previously tracked performance statistics.
 */
/**
 * @def PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
 * @brief Defines the `PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Adds the provided entry to the plssvm::detail::performance_tracker singleton if tracking is currently enabled.
 */
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)

    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_PAUSE() \
        plssvm::detail::performance_tracker::instance().pause_tracking()

    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_RESUME() \
        plssvm::detail::performance_tracker::instance().resume_tracking()

    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE() \
        plssvm::detail::performance_tracker::instance().save(PLSSVM_PERFORMANCE_TRACKER_OUTPUT_FILE)

    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(entry)                \
        if (plssvm::detail::performance_tracker::instance().is_tracking()) {           \
            plssvm::detail::performance_tracker::instance().add_tracking_entry(entry); \
        }
#else

    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_PAUSE()
    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_RESUME()
    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE()
    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(entry)

#endif

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_PERFORMANCE_TRACKER_HPP_
