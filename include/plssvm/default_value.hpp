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

#ifndef PLSSVM_DEFAULT_VALUE_HPP_
#define PLSSVM_DEFAULT_VALUE_HPP_
#pragma once

#include "plssvm/detail/type_traits.hpp"  // PLSSVM_REQUIRES

#include "fmt/ostream.h"  // fmt::formatter, fmt::ostream_formatter

#include <cstddef>      // std::size_t
#include <functional>   // std::hash
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <type_traits>  // std::is_nothrow_default_constructible_v, std::is_nothrow_move_constructible_v, std::is_nothrow_move_assignable_v, std::is_nothrow_swappable_v,
                        // std::is_convertible_v, std::true_type, std::false_type, std::remove_reference_t, std::remove_cv_t
#include <utility>      // std::move_if_noexcept, std::swap

namespace plssvm {

//*************************************************************************************************************************************//
//                                                            default_init                                                             //
//*************************************************************************************************************************************//

/**
 * @brief This class denotes an explicit default value initialization used to distinguish between the default value or user provided initialization in the `default_value` class.
 * @tparam T the type of the default value
 */
template <typename T>
struct default_init {
    /**
     * @brief Default construct the default initialization value.
     */
    constexpr default_init() noexcept(std::is_nothrow_default_constructible_v<T>) = default;
    /**
     * @brief Set the initialization value to @p val.
     * @param[in] val the explicit default initialization value
     */
    constexpr explicit default_init(T val) noexcept(std::is_nothrow_move_constructible_v<T>) :
        value{ std::move_if_noexcept(val) } {}

    /// The explicit default initialization value.
    T value{};
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
    /// @cond Doxygen_suppress
    // befriend a default_value encapsulating another type (used to be able to convert between, e.g., default_value<int> and default_value<float>)
    template <typename>
    friend class default_value;
    // befriend input stream operator to be able to construct a default_value from a std::istream
    template <typename U>
    friend std::istream &operator>>(std::istream &in, default_value<U> &);
    /// @endcond

  public:
    /// The type encapsulated by this default_value.
    using value_type = T;

    /**
     * @brief Copy construct a default_value object using the provided **default** value.
     * @details Afterward, `is_default()` will return `true`!
     * @param[in] default_val set the default value of this default_value wrapper
     */
    constexpr explicit default_value(default_init<value_type> default_val = default_init<value_type>{}) noexcept(std::is_nothrow_move_constructible_v<value_type>) :
        default_init_{ std::move_if_noexcept(default_val) } {}
    /**
     * @brief **Override** the previously provided default value with the new, non-default value.
     * @details Afterward, `is_default()` will return `false`!
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
    template <typename U, PLSSVM_REQUIRES(std::is_convertible_v<U, value_type>)>
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
    template <typename U, PLSSVM_REQUIRES(std::is_convertible_v<U, value_type>)>
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
    template <typename U, PLSSVM_REQUIRES(std::is_convertible_v<U, value_type>)>
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
    template <typename U, PLSSVM_REQUIRES(std::is_convertible_v<U, value_type>)>
    constexpr default_value &operator=(default_value<U> &&other) noexcept(std::is_nothrow_move_assignable_v<U>) {
        value_ = static_cast<value_type>(std::move_if_noexcept(other.value_));
        use_default_init_ = other.use_default_init_;
        default_init_ = default_init<value_type>{ static_cast<value_type>(std::move_if_noexcept(other.default_init_.value)) };
        return *this;
    }

    /**
     * @brief Get the currently active value: the user provided value if provided, otherwise the default value is returned.
     * @return the active value (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr const value_type &value() const noexcept {
        return this->is_default() ? default_init_.value : value_;
    }
    /**
     * @copydoc plssvm::default_value::value()
     */
    [[nodiscard]] constexpr operator const value_type &() const noexcept {
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
inline std::ostream &operator<<(std::ostream &out, const default_value<T> &val) {
    return out << val.value();
}
/**
 * @brief Use the input-stream @p in to initialize the default_value @p val.
 * @details Sets the user defined value, i.e., `plssvm::default_value::is_default()` will return `false` and the default value will **not** be used.
 * @tparam T the type of the wrapped value
 * @param[in,out] in input-stream to extract the wrapped default_value value from
 * @param[in] val the default_value
 * @return the input-stream
 */
template <typename T>
inline std::istream &operator>>(std::istream &in, default_value<T> &val) {
    in >> val.value_;
    val.use_default_init_ = false;
    return in;
}

/**
 * @brief Swap the content of two default_values @p lhs and @p rhs.
 * @param[in,out] lhs the first default_value
 * @param[in,out] rhs the second default_value
 */
template <typename T>
constexpr void swap(default_value<T> &lhs, default_value<T> &rhs) noexcept(noexcept(lhs.swap(rhs))) {
    lhs.swap(rhs);
}

/**
 * @brief Compares **the two active** values @p lhs and @p rhs for equality.
 * @tparam T the type of the wrapped value
 * @param[in] lhs the first default_value
 * @param[in] rhs the second default_value
 * @return `true` if the active values are equal, `false` otherwise (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr bool operator==(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return lhs.value() == rhs.value();
}
/**
 * @copydoc operator==(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator==(const default_value<T> &lhs, const T &rhs) noexcept {
    return lhs.value() == rhs;
}
/**
 * @copydoc operator==(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator==(const T &lhs, const default_value<T> &rhs) noexcept {
    return lhs == rhs.value();
}

/**
 * @brief Compares **the two active** values @p lhs and @p rhs for inequality.
 * @tparam T the type of the wrapped value
 * @param[in] lhs the first default_value
 * @param[in] rhs the second default_value
 * @return `true` if the active values are unequal, `false` otherwise (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr bool operator!=(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return !(lhs == rhs);
}
/**
 * @copydoc operator!=(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator!=(const default_value<T> &lhs, const T &rhs) noexcept {
    return !(lhs == rhs);
}
/**
 * @copydoc operator!=(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator!=(const T &lhs, const default_value<T> &rhs) noexcept {
    return !(lhs == rhs);
}

/**
 * @brief Compares **the two active** values: @p lhs < @p rhs.
 * @tparam T the type of the wrapped value
 * @param[in] lhs the first default_value
 * @param[in] rhs the second default_value
 * @return `true` if the active values of @p lhs is less than the active value of @p rhs, `false` otherwise (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr bool operator<(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return lhs.value() < rhs.value();
}
/**
 * @copydoc operator<(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator<(const default_value<T> &lhs, const T &rhs) noexcept {
    return lhs.value() < rhs;
}
/**
 * @copydoc operator<(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator<(const T &lhs, const default_value<T> &rhs) noexcept {
    return lhs < rhs.value();
}

/**
 * @brief Compares **the two active** values: @p lhs > @p rhs.
 * @tparam T the type of the wrapped value
 * @param[in] lhs the first default_value
 * @param[in] rhs the second default_value
 * @return `true` if the active values of @p lhs is greater than the active value of @p rhs, `false` otherwise (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr bool operator>(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return rhs < lhs;
}

/**
 * @copydoc operator>(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator>(const default_value<T> &lhs, const T &rhs) noexcept {
    return lhs.value() > rhs;
}

/**
 * @copydoc operator>(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator>(const T &lhs, const default_value<T> &rhs) noexcept {
    return lhs > rhs.value();
}

/**
 * @brief Compares **the two active** values: @p lhs <= @p rhs.
 * @tparam T the type of the wrapped value
 * @param[in] lhs the first default_value
 * @param[in] rhs the second default_value
 * @return `true` if the active values of @p lhs is less or equal than the active value of @p rhs, `false` otherwise (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr bool operator<=(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return !(lhs > rhs);
}
/**
 * @copydoc operator<=(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator<=(const default_value<T> &lhs, const T &rhs) noexcept {
    return !(lhs.value() > rhs);
}
/**
 * @copydoc operator<=(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator<=(const T &lhs, const default_value<T> &rhs) noexcept {
    return !(lhs > rhs.value());
}

/**
 * @brief Compares **the two active** values: @p lhs >= @p rhs.
 * @tparam T the type of the wrapped value
 * @param[in] lhs the first default_value
 * @param[in] rhs the second default_value
 * @return `true` if the active values of @p lhs is greater or equal than the active value of @p rhs, `false` otherwise (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] constexpr bool operator>=(const default_value<T> &lhs, const default_value<T> &rhs) noexcept {
    return !(lhs < rhs);
}
/**
 * @copydoc operator>=(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator>=(const default_value<T> &lhs, const T &rhs) noexcept {
    return !(lhs.value() < rhs);
}
/**
 * @copydoc operator>=(const default_value<T> &, const default_value<T> &)
 */
template <typename T>
[[nodiscard]] constexpr bool operator>=(const T &lhs, const default_value<T> &rhs) noexcept {
    return !(lhs < rhs.value());
}

/// @cond Doxygen_suppress
namespace detail {

/**
 * @brief Test whether the given type @p T is of type `plssvm::default_value` (represents the `false` case).
 * @details Inherits from `std::false_type`.
 * @tparam T the type to check whether it is a `plssvm::default_value`
 */
template <typename T>
struct is_default_value : std::false_type {};
/**
 * @brief Test whether the given type @p T is of type `plssvm::default_value` (represents the `true` case).
 * @details Inherits from `std::true_type`.
 * @tparam T the type to check whether it is a `plssvm::default_value`
 */
template <typename T>
struct is_default_value<default_value<T>> : std::true_type {};

}  // namespace detail
/// @endcond

/**
 * @brief Test whether the given type @p T is of type `plssvm::default_value` ignoring all top-level const, volatile, and reference qualifiers.
 * @tparam T the type to check whether it is a `plssvm::default_value`
 */
template <typename T>
struct is_default_value : detail::is_default_value<std::remove_cv_t<std::remove_reference_t<T>>> {};  // can't use detail::remove_cvref_t because of circular dependencies
/**
 * @brief Test whether the given type @p T is of type `plssvm::default_value` ignoring all top-level const, volatile, and reference qualifiers.
 */
template <typename T>
constexpr bool is_default_value_v = is_default_value<T>::value;

}  // namespace plssvm

namespace std {

/**
 * @brief Hashing struct specialization in the `std` namespace for a default_value.
 * @details Necessary to be able to use a default_value, e.g., in a `std::unordered_map`.
 */
template <typename T>
struct hash<plssvm::default_value<T>> {
    /**
     * @brief Overload the function call operator for a default_value.
     * @details Hashes the currently active value of @p val using its default hash function.
     * @param[in] val the default_value to hash
     * @return the hash value of @p val
     */
    std::size_t operator()(const plssvm::default_value<T> &val) const noexcept {
        return std::hash<T>{}(val.value());
    }
};

}  // namespace std

template <typename T>
struct fmt::formatter<plssvm::default_value<T>> : fmt::ostream_formatter {};

#endif  // PLSSVM_DEFAULT_VALUE_HPP_