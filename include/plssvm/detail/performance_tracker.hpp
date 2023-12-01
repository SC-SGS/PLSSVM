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

#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::remove_cvref_t
#include "plssvm/detail/utility.hpp"      // PLSSVM_EXTERN
#include "plssvm/parameter.hpp"           // plssvm::parameter

#include "plssvm/detail/cmd/parser_predict.hpp"  // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"    // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"    // plssvm::detail::cmd::parser_train

#include "fmt/chrono.h"  // format std::chrono types
#include "fmt/format.h"  // fmt::format, fmt::join, fmt::formatter

#include <map>          // std::map
#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <type_traits>  // std::false_type, std::true_type
#include <utility>      // std::move
#include <vector>       // std::vector

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
    const std::string entry_name{};
    /// The tracked value in the YAML file.
    const T entry_value{};
};

}  // namespace plssvm::detail

/**
 * @brief Custom tracking_entry formatter to be able to use fmt format specifiers for values of type T.
 * @tparam T the performance tracked type
 */
template <typename T>
struct fmt::formatter<plssvm::detail::tracking_entry<T>> : fmt::formatter<T> {
    /**
     * @brief Format the tracking @p entry using the provided format specifier for type T.
     * @tparam FormatContext the type of the format context
     * @param[in] entry the tracking entry to format
     * @param[in,out] context the format context
     * @return the formatted string
     */
    template <typename FormatContext>
    auto format(const plssvm::detail::tracking_entry<T> &entry, FormatContext &context) {
        return fmt::formatter<T>::format(entry.entry_value, context);
    }
};

namespace plssvm::detail {

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
     * @breif Default constructor. **Must** be implemented in .cpp file.
     */
    performance_tracker();
    /**
     * @brief Default destructor. **Must** be implemented in .cpp file.
     */
    ~performance_tracker();

    /**
     * @brief Add a tracking_entry to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     * @tparam T the type of the value the tracking_entry @p entry encapsulates
     * @param[in] entry the entry to add
     */
    template <typename T>
    void add_tracking_entry(const tracking_entry<T> &entry);
    /**
     * @brief Add a tracking_entry consisting of multiple values stored in a `std::vector` to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     * @tparam T the type of the value the tracking_entry @p entry encapsulates
     * @param[in] entry the `std::vector` entry to add
     */
    template <typename T>
    void add_tracking_entry(const tracking_entry<std::vector<T>> &entry);
    /**
     * @brief Add a tracking_entry encapsulating a plssvm::parameter to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the plssvm::parameter as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<plssvm::parameter> &entry);
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
     *          If @p filename is empty, dumps the tracking entries to `std::cout` instead.
     * @param[in] filename the file to add the performance tracking results to
     */
    void save(const std::string &filename);
    /**
     * @brief Write all stored tracking entries to the output stream @p out.
     * @param[in] out the output stream to write the performance tracking results to
     */
    void save(std::ostream &out);

    /**
     * @brief Pause the current tracking, i.e., all calls to `add_tracking_entry` do nothing.
     */
    void pause_tracking() noexcept;
    /**
     * @brief Resume the tracking, i.e., all calls to `add_tracking_entry` do track the values again.
     */
    void resume_tracking() noexcept;
    /**
     * @brief Check whether tracking is currently active or paused.
     * @return `true` if tracking is enabled, `false` if it is currently paused (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_tracking() const noexcept;

    /**
     * @brief Return the currently available tracking entries.
     * @return the previously added tracking entries (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::map<std::string, std::map<std::string, std::vector<std::string>>> &get_tracking_entries() noexcept;
    /**
     * @brief Remove all currently saved tracking entries from the performance tracker.
     */
    void clear_tracking_entries() noexcept;

  private:
    /// All performance statistics grouped by their specified categories.
    std::map<std::string, std::map<std::string, std::vector<std::string>>> tracking_statistics_{};
    /// The tracking is enabled by default.
    bool is_tracking_{ true };
};

template <typename T>
void performance_tracker::add_tracking_entry(const tracking_entry<T> &entry) {
    if (is_tracking()) {
        const std::string entry_value_str = fmt::format(std::is_same_v<T, std::string> ? "\"{}\"" : "{}", entry.entry_value);

        if (detail::contains(tracking_statistics_, entry.entry_category)) {
            // category already exists -> check if entry already exists
            if (detail::contains(tracking_statistics_[entry.entry_category], entry.entry_name)) {
                // entry already exists -> add new value to this entry's list
                tracking_statistics_[entry.entry_category][entry.entry_name].push_back(entry_value_str);
            } else {
                // entry does not exist -> create new entry and add value as initial value to the vector
                tracking_statistics_[entry.entry_category].emplace(entry.entry_name, std::vector<std::string>{ entry_value_str });
            }
        } else {
            // category does not exist -> create new category with the current entry + entry value
            tracking_statistics_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{ { entry.entry_name, std::vector<std::string>{ entry_value_str } } });
        }
    }
}

template <typename T>
void performance_tracker::add_tracking_entry(const tracking_entry<std::vector<T>> &entry) {
    if (is_tracking()) {
        std::string entry_value_str;
        if constexpr (std::is_same_v<T, std::string>) {
            entry_value_str = fmt::format("[\"{}\"]", fmt::join(entry.entry_value, "\", \""));
        } else {
            entry_value_str = fmt::format("[{}]", fmt::join(entry.entry_value, ", "));
        }

        if (detail::contains(tracking_statistics_, entry.entry_category)) {
            // category already exists -> check if entry already exists
            if (detail::contains(tracking_statistics_[entry.entry_category], entry.entry_name)) {
                // entry already exists -> add new value to this entry's list
                tracking_statistics_[entry.entry_category][entry.entry_name].push_back(entry_value_str);
            } else {
                // entry does not exist -> create new entry and add value as initial value to the vector
                tracking_statistics_[entry.entry_category].emplace(entry.entry_name, std::vector<std::string>{ entry_value_str });
            }
        } else {
            // category does not exist -> create new category with the current entry + entry value
            tracking_statistics_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{ { entry.entry_name, std::vector<std::string>{ entry_value_str } } });
        }
    }
}

/// The global performance tracker instance used for the default tracking.
PLSSVM_EXTERN std::shared_ptr<performance_tracker> global_tracker;

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
        ::plssvm::detail::global_tracker->pause_tracking()

    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_RESUME() \
        ::plssvm::detail::global_tracker->resume_tracking()

    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE(filename) \
        ::plssvm::detail::global_tracker->save(filename)

    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(entry) \
        ::plssvm::detail::global_tracker->add_tracking_entry(entry)
#else

    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_PAUSE()
    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_RESUME()
    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_SAVE(filename)
    #define PLSSVM_DETAIL_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(entry)

#endif

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_PERFORMANCE_TRACKER_HPP_
