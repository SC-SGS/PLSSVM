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

#ifndef PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_HPP_
#define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_HPP_
#pragma once

#include "plssvm/detail/cmd/parser_predict.hpp"         // plssvm::detail::cmd::parser_predict
#include "plssvm/detail/cmd/parser_scale.hpp"           // plssvm::detail::cmd::parser_scale
#include "plssvm/detail/cmd/parser_train.hpp"           // plssvm::detail::cmd::parser_train
#include "plssvm/detail/memory_size.hpp"                // plssvm::detail::memory_size
#include "plssvm/detail/tracking/events.hpp"            // plssvm::detail::tracking::{events, event}
#include "plssvm/detail/tracking/hardware_sampler.hpp"  // plssvm::detail::tracking::hardware_sampler
#include "plssvm/detail/type_traits.hpp"                // plssvm::detail::remove_cvref_t
#include "plssvm/detail/utility.hpp"                    // PLSSVM_EXTERN
#include "plssvm/parameter.hpp"                         // plssvm::parameter

#include "fmt/chrono.h"   // format std::chrono types
#include "fmt/core.h"     // fmt::format, fmt::formatter
#include "fmt/format.h"   // fmt::join,
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <chrono>       // std::chrono::system_clock::time_point
#include <map>          // std::map
#include <ostream>      // std::ostream
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <type_traits>  // std::false_type, std::true_type, std::is_same_v
#include <utility>      // std::move, std::pair
#include <vector>       // std::vector

namespace plssvm::detail::tracking {

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
        entry_category{ category },
        entry_name{ name },
        entry_value{ std::move(value) } { }

    /**
     * @brief Explicitly delete copy-constructor.
     */
    tracking_entry(const tracking_entry &) = delete;
    /**
     * @brief Explicitly delete move-constructor.
     */
    tracking_entry(tracking_entry &&) noexcept = delete;
    /**
     * @brief Explicitly delete copy-assignment operator.
     */
    tracking_entry &operator=(const tracking_entry &) = delete;
    /**
     * @brief Explicitly delete move-assignment operator.
     */
    tracking_entry &operator=(tracking_entry &&) noexcept = delete;
    /**
     * @brief Explicitly default destructor.
     */
    ~tracking_entry() = default;

    /// The category to which this tracking entry belongs; used for grouping in the resulting YAML file.
    const std::string entry_category{};
    /// The name of the tracking entry displayed in the YAML file.
    const std::string entry_name{};
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
struct is_tracking_entry : std::false_type { };

/**
 * @brief Sets the `value` to `true` since it **is** a tracking entry.
 */
template <typename T>
struct is_tracking_entry<tracking_entry<T>> : std::true_type { };

}  // namespace impl

/**
 * @brief Check whether @p T is a tracking entry. Ignores all top-level const, volatile, and reference qualifiers.
 * @tparam T the type to check whether it is a tracking entry or not
 */
template <typename T>
struct is_tracking_entry : impl::is_tracking_entry<detail::remove_cvref_t<T>> { };

/**
 * @copydoc plssvm::detail::is_tracking_entry
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
     * @brief Default constructor. **Must** be implemented in .cpp file.
     */
    performance_tracker();
    /**
     * @brief Default copy-constructor. **Must** be implemented in .cpp file.
     */
    performance_tracker(const performance_tracker &);
    /**
     * @brief Default move-constructor. **Must** be implemented in .cpp file.
     */
    performance_tracker(performance_tracker &&) noexcept;
    /**
     * @brief Default copy-assignment operator. **Must** be implemented in .cpp file.
     * @return `*this`
     */
    performance_tracker &operator=(const performance_tracker &);
    /**
     * @brief Default move-assignment operator. **Must** be implemented in .cpp file.
     * @return `*this`
     */
    performance_tracker &operator=(performance_tracker &&) noexcept;
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
     * @details Saves a string containing the entry name and values as `[0, 1, 2]` in a map with the entry category as key.
     * @tparam T the type of the value the tracking_entry @p entry encapsulates
     * @param[in] entry the `std::vector` entry to add
     */
    template <typename T>
    void add_tracking_entry(const tracking_entry<std::vector<T>> &entry);
    /**
     * @brief Add a tracking_entry encapsulating a `plssvm::parameter` to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the `plssvm::parameter` as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<plssvm::parameter> &entry);
    /**
     * @brief Add a tracking_entry encapsulating a `plssvm::detail::cmd::parser_train` to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the `plssvm::detail::cmd::parser_train` as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<cmd::parser_train> &entry);
    /**
     * @brief Add a tracking_entry encapsulating a `plssvm::detail::cmd::parser_predict` to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the `plssvm::detail::cmd::parser_predict` as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<cmd::parser_predict> &entry);
    /**
     * @brief Add a tracking_entry encapsulating a `plssvm::detail::cmd::parser_scale` to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Adds all values stored in the `plssvm::detail::cmd::parser_scale` as tracking entries.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<cmd::parser_scale> &entry);
    /**
     * @brief Add a tracking_entry encapsulating a `plssvm::detail::tracking::hardware_tracker` (or any of its subclasses) and the hardware sampling start time to this performance tracker.
     * @details Saves a string containing the entry name and value in a map with the entry category as key.
     *          Uses the `plssvm::detail::tracking::hardware_tracker::generate_yaml_string` function to generate the tracking entry.
     * @param[in] entry the entry to add
     */
    void add_tracking_entry(const tracking_entry<std::pair<hardware_sampler *, std::chrono::system_clock::time_point>> &entry);

    /**
     * @brief Add an event to this hardware sampler.
     * @param[in] name the event name
     */
    void add_event(std::string name);

    /**
     * @brief Write all stored tracking entries to the [YAML](https://yaml.org/) file @p filename.
     * @details Appends all entries at the end of the file creating a new YAML document.
     *          If @p filename is empty, dumps the tracking entries to `std::clog` instead.
     * @note They tracking entries are always output on the console even if the current `plssvm::verbosity_level` is `quiet`!
     * @param[in] filename the file to add the performance tracking results to
     */
    void save(const std::string &filename);
    /**
     * @brief Write all stored tracking entries to the output stream @p out.
     * @note They tracking entries are always output on the console even if the current `plssvm::verbosity_level` is `quiet`!
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
     * @details Grouped as a map containing the categories as a string and a map containing the tracking entry names and all associated values.
     * @return the previously added tracking entries (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::map<std::string, std::map<std::string, std::vector<std::string>>> &get_tracking_entries() noexcept;
    /**
     * @brief Remove all currently saved tracking entries from the performance tracker.
     */
    void clear_tracking_entries() noexcept;

  private:
    /// All tracking entries grouped by their specified categories.
    std::map<std::string, std::map<std::string, std::vector<std::string>>> tracking_entries_{};
    /// All special events mainly used for hardware sampling.
    events events_{};
    /// Flag indicating whether tracking is currently enabled or disabled. Tracking is enabled by default.
    bool is_tracking_{ true };
};

template <typename T>
void performance_tracker::add_tracking_entry(const tracking_entry<T> &entry) {
    // check whether entries should currently be tracked
    if (this->is_tracking()) {
        std::string entry_value_str{};
        if constexpr (std::is_same_v<T, std::string>) {
            // escape strings with "" since they may contain whitespaces
            entry_value_str = fmt::format("\"{}\"", entry.entry_value);
        } else if constexpr (std::is_same_v<T, detail::memory_size>) {
            // dump the memory size in BYTES to the file (to get rid of the hard to parse memory_size units)
            entry_value_str = fmt::format("{}", entry.entry_value.num_bytes());
        } else {
            entry_value_str = fmt::format("{}", entry.entry_value);
        }

        if (detail::contains(tracking_entries_, entry.entry_category)) {
            // category already exists -> check if entry already exists
            if (detail::contains(tracking_entries_[entry.entry_category], entry.entry_name)) {
                // entry already exists -> add new value to this entry's list
                tracking_entries_[entry.entry_category][entry.entry_name].push_back(std::move(entry_value_str));
            } else {
                // entry does not exist -> create new entry and add value as initial value to the vector
                tracking_entries_[entry.entry_category].emplace(entry.entry_name, std::vector<std::string>{ std::move(entry_value_str) });
            }
        } else {
            // category does not exist -> create new category with the current entry + entry value
            tracking_entries_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{ { entry.entry_name, std::vector<std::string>{ std::move(entry_value_str) } } });
        }
    }
}

template <typename T>
void performance_tracker::add_tracking_entry(const tracking_entry<std::vector<T>> &entry) {
    // check whether entries should currently be tracked
    if (this->is_tracking()) {
        std::string entry_value_str{};
        if constexpr (std::is_same_v<T, std::string>) {
            // escape strings with "" since they may contain whitespaces
            entry_value_str = fmt::format("[\"{}\"]", fmt::join(entry.entry_value, "\", \""));
        } else if constexpr (std::is_same_v<T, detail::memory_size>) {
            // dump all memory sizes in BYTES to the file (to get rid of the hard to parse memory_size units)
            std::vector<unsigned long long> byte_values{};
            byte_values.reserve(entry.entry_value.size());
            for (const memory_size mem : entry.entry_value) {
                byte_values.push_back(mem.num_bytes());
            }
            entry_value_str = fmt::format("[{}]", fmt::join(byte_values, ", "));
        } else {
            entry_value_str = fmt::format("[{}]", fmt::join(entry.entry_value, ", "));
        }

        if (detail::contains(tracking_entries_, entry.entry_category)) {
            // category already exists -> check if entry already exists
            if (detail::contains(tracking_entries_[entry.entry_category], entry.entry_name)) {
                // entry already exists -> add new value to this entry's list
                tracking_entries_[entry.entry_category][entry.entry_name].push_back(std::move(entry_value_str));
            } else {
                // entry does not exist -> create new entry and add value as initial value to the vector
                tracking_entries_[entry.entry_category].emplace(entry.entry_name, std::vector<std::string>{ std::move(entry_value_str) });
            }
        } else {
            // category does not exist -> create new category with the current entry + entry value
            tracking_entries_.emplace(entry.entry_category, std::map<std::string, std::vector<std::string>>{ { entry.entry_name, std::vector<std::string>{ std::move(entry_value_str) } } });
        }
    }
}

/// The global performance tracker singleton function used for the default tracking.
performance_tracker &global_performance_tracker();

/**
 * @def PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_PAUSE
 * @brief Defines the `PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_PAUSE` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Pause the tracking functionality if performance tracking has been enabled during the CMake configuration.
 */
/**
 * @def PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_RESUME
 * @brief Defines the `PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_RESUME` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Resume the tracking functionality if performance tracking has been enabled during the CMake configuration.
 */
/**
 * @def PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE
 * @brief Defines the `PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Save the previously tracked entries if performance tracking has been enabled during the CMake configuration.
 */
/**
 * @def PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY
 * @brief Defines the `PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY` macro if `PLSSVM_PERFORMANCE_TRACKER_ENABLED` is defined.
 * @details Adds the provided entry to the `plssvm::detail::tracking::performance_tracker` singleton if performance tracking has been enabled during the CMake configuration.
 */
#if defined(PLSSVM_PERFORMANCE_TRACKER_ENABLED)

    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_PAUSE() \
        ::plssvm::detail::tracking::global_performance_tracker().pause_tracking()

    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_RESUME() \
        ::plssvm::detail::tracking::global_performance_tracker().resume_tracking()

    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(filename) \
        ::plssvm::detail::tracking::global_performance_tracker().save(filename)

    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(entry) \
        ::plssvm::detail::tracking::global_performance_tracker().add_tracking_entry(entry)

    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT(event_name) \
        ::plssvm::detail::tracking::global_performance_tracker().add_event(event_name)

#else

    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_PAUSE()
    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_RESUME()
    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_SAVE(filename)
    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_TRACKING_ENTRY(entry)
    #define PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_ADD_EVENT(event_name)

#endif

}  // namespace plssvm::detail::tracking

template <typename T>
struct fmt::formatter<plssvm::detail::tracking::tracking_entry<T>> : fmt::ostream_formatter { };

#endif  // PLSSVM_DETAIL_TRACKING_PERFORMANCE_TRACKER_HPP_
