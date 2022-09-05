/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a class used to be able to distinguish between the default value of a variable and an assigned value.
 */

#pragma once

#include <cstddef>      // std::size_t
#include <functional>   // std::hash
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <type_traits>  // std::is_nothrow_default_constructible_v, std::is_nothrow_move_constructible_v, std::is_nothrow_move_assignable_v, std::is_nothrow_swappable_v,
                        // std::enable_if_t, std::is_convertible_v
#include <utility>      // std::move_if_noexcept, std::swap

namespace plssvm {

//*************************************************************************************************************************************//
//                                                            default_init                                                             //
//*************************************************************************************************************************************//

/**
 * @brief This class denotes an explicit default value used to distinguish between the default value or user provided value in the `default_value`class.
 * @tparam T the type of the default value
 */
template <typename T>
struct default_init {
    /**
     * @brief Default construct the default value.
     */
    constexpr default_init() noexcept(std::is_nothrow_default_constructible_v<T>) = default;
    /**
     * @brief Set the default value to @p val.
     * @param[in] val the explicit default value
     */
    constexpr explicit default_init(T val) noexcept(std::is_nothrow_move_constructible_v<T>) :
        value{ std::move_if_noexcept(val) } {}

    /// The explicit default value.
    T value;
};

//*************************************************************************************************************************************//
//                                                            default_value                                                            //
//*************************************************************************************************************************************//

/**
 * @brief This class encapsulates a value that may be a default value or not.
 * @tparam T the type of the encapsulated value
 */
template <typename T>
class default_value {
    // befriend a default_value encapsulating another type (used to be able to convert between, e.g., default_value<int> and default_value<float>)
    template <typename>
    friend class default_value;
    // befriend input stream operator to be able to construct a default_value from a std::istream
    template <typename U>
    friend std::istream &operator>>(std::istream &in, default_value<U> &);

  public:
    /// The type encapsulated by this default_value.
    using value_type = T;

    /**
     * @brief Construct a default_value object using the provided **default** value.
     * @note `is_default()` will return `true` afterward!
     */
    constexpr explicit default_value(default_init<value_type> default_val = default_init<value_type>{}) noexcept(std::is_nothrow_move_constructible_v<value_type>) :
        default_init_{ std::move_if_noexcept(default_val) } {}
    /**
     * @brief **Override** the previously provided default value with the new, non-default value.
     * @param[in] non_default_val the non-default value
     * @return `*this`
     */
    constexpr default_value &operator=(value_type non_default_val) noexcept(std::is_nothrow_move_assignable_v<value_type>) {
        value_ = std::move_if_noexcept(non_default_val);
        use_default_init_ = false;
        return *this;
    }

    /**
     * @brief Copy-construct a new default_value from a possibly other type.
     * @tparam U the type of the other default_value wrapper
     * @param[in] other the other default_value potentially wrapping another type
     * @note `U` must be convertible to `value_type`!
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    constexpr explicit default_value(const default_value<U> &other) noexcept(std::is_nothrow_copy_constructible_v<U>) :
        value_{ static_cast<value_type>(other.value_) },
        use_default_init_{ other.use_default_init_ },
        default_init_{ static_cast<value_type>(other.default_init_.value) } {}
    /**
     * @brief Move-construct a new default_value_from a possibly other type.
     * @tparam U the type of the other default_value wrapper
     * @param[in,out] other the other default_value potentially wrapping another type
     * @note `U` must be convertible to `value_type`!
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    constexpr explicit default_value(default_value<U> &&other) noexcept(std::is_nothrow_move_constructible_v<U>) :
        value_{ static_cast<value_type>(std::move_if_noexcept(other.value_)) },
        use_default_init_{ other.use_default_init_ },
        default_init_{ static_cast<value_type>(std::move_if_noexcept(other.default_init_.value)) } {}
    /**
     * @brief Copy-assign a new default_value from a possible other type.
     * @tparam U the type of the other default_value wrapper
     * @param other the other default_value potentially wrapping another type
     * @note `U` must be convertible to `value_type`!
     * @return `*this`
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    constexpr default_value &operator=(const default_value<U> &other) noexcept(std::is_nothrow_copy_assignable_v<U>) {
        value_ = static_cast<value_type>(other.value_);
        use_default_init_ = other.use_default_init_;
        default_init_ = default_init<value_type>{ static_cast<value_type>(other.default_init_.value) };
        return *this;
    }
    /**
     * @brief Move-assign a new default_value from a possible other type.
     * @tparam U the type of the other default_value wrapper
     * @param other the other default_value potentially wrapping another type
     * @note `U` must be convertible to `value_type`!
     * @return `*this`
     */
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    constexpr default_value &operator=(default_value<U> &&other) noexcept(std::is_nothrow_move_assignable_v<U>) {
        value_ = static_cast<value_type>(std::move_if_noexcept(other.value_));
        use_default_init_ = other.use_default_init_;
        default_init_ = default_init<value_type>{ static_cast<value_type>(std::move_if_noexcept(other.default_init_.value)) };
        return *this;
    }

    /**
     * @brief Get the currently active value: the user provided value if provided, else the default value.
     * @return the active value (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr const value_type &value() const noexcept {
        return this->is_default() ? default_init_.value : value_;
    }
    /**
     * @brief Get the currently active value: the user provided value if provided, else the default value.
     * @return the active value (`[[nodiscard]]`)
     */
    [[nodiscard]] operator const value_type &() const noexcept {
        return this->value();
    }
    /**
     * @brief Get the default value even if it has already been overwritten by the user.
     * @return the active value (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr const value_type &get_default() const noexcept {
        return default_init_.value;
    }
    /**
     * @brief Check whether the currently active value is the default value.
     * @return `true` if the default value is currently active, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr bool is_default() const noexcept {
        return use_default_init_;
    }

    /**
     * @brief Swap the content of two default_values.
     * @param[in,out] other the other default_value
     */
    constexpr void swap(default_value &other) noexcept(std::is_nothrow_swappable_v<T>) {
        using std::swap;
        swap(value_, other.value_);
        swap(use_default_init_, other.use_default_init_);
        swap(default_init_, other.default_init_);
    }
    /**
     * @brief Set the current active value back to the default value.
     */
    constexpr void reset() noexcept {
        use_default_init_ = true;
    }

  private:
    /// The wrapped value to be used if `use_default_init_` is `false`.
    value_type value_{};
    /// Flag used to determine whether the default value should be used or the user defined value.
    bool use_default_init_{ true };
    /// The wrapped default value used if `use_default_init_` is `true`.
    default_init<value_type> default_init_{};
};

/**
 * @brief Output the wrapped value of @p val to the given output-stream @p out.
 * @tparam T the type of the wrapped value
 * @param[in,out] out the output-stream to write the wrapped default_value value to
 * @param[in] val the default_value
 * @return the output-stream
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, const default_value<T> &val) {
    return out << val.value();
}
/**
 * @brief Use the input-stream @p in to initialize the default_value @p val.
 * @details Sets the user defined value, i.e., `is_default()` will return `false` and the default value will **not** be used.
 * @param[in,out] in input-stream to extract the wrapped default_value value from
 * @param[in] val the default_value
 * @return the input-stream
 */
template <typename T>
std::istream &operator>>(std::istream &in, default_value<T> &val) {
    in >> val.value_;
    val.use_default_init_ = false;
    return in;
}

/**
 * @brief Swap the content of two default_values @p lhs and @p rhs.
 * @param[in,out] lhs the first default_value
 * @param[in,out] lhs the second default_value
 */
template <typename T>
constexpr void swap(default_value<T> &lhs, default_value<T> &rhs) noexcept(noexcept(lhs.swap(rhs))) {
    lhs.swap(rhs);
}

// comparison operations
template <typename T>
constexpr bool operator==(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return lhs.value() == rhs.value();
}
template <typename T>
constexpr bool operator!=(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return !(lhs == rhs);
}
template <typename T>
constexpr bool operator<(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return lhs.value() < rhs.value();
}
template <typename T>
constexpr bool operator>(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return rhs < lhs;
}
template <typename T>
constexpr bool operator<=(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return !(lhs > rhs);
}
template <typename T>
constexpr bool operator>=(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return !(lhs < rhs);
}

}  // namespace plssvm

namespace std {

template <typename T>
struct hash<plssvm::default_value<T>> {
    std::size_t operator()(const plssvm::default_value<T> &val) const noexcept {
        return std::hash<T>{}(val.value());
    }
};

}  // namespace std