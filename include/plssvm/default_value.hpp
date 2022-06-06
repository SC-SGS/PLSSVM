#pragma once

#include <cstddef>      // std::size_t
#include <functional>   // std::hash
#include <istream>      // std::istream
#include <ostream>      // std::ostream
#include <type_traits>  // std::is_nothrow_default_constructible_v, std::is_nothrow_move_constructible_v, std::is_nothrow_move_assignable_v, std::is_nothrow_swappable_v,
                        // std::enable_if_t, std::is_convertible_v
#include <utility>      // std::move_if_noexcept, std::swap


namespace plssvm {


template <typename T>
struct default_init {
    constexpr default_init() noexcept(std::is_nothrow_default_constructible_v<T>) = default;
    constexpr explicit default_init(T val) noexcept(std::is_nothrow_move_constructible_v<T>) : value{ std::move_if_noexcept(val) } { }

    T value;
};

template <typename T>
class default_value {
    template <typename>
    friend class default_value;
    template <typename U>
    friend std::istream &operator>>(std::istream &in, default_value<U> &);
  public:
    using value_type = T;

    constexpr default_value() noexcept(std::is_nothrow_default_constructible_v<value_type>) = default;
    constexpr default_value(default_init<value_type> default_val) noexcept(std::is_nothrow_move_constructible_v<value_type>) : default_init_{ std::move_if_noexcept(default_val) } { }
    constexpr default_value &operator=(value_type non_default_val) noexcept(std::is_nothrow_move_assignable_v<value_type>) {
        value_ = std::move_if_noexcept(non_default_val);
        use_default_init_ = false;
        return *this;
    }
    template <typename U, std::enable_if_t<std::is_convertible_v<U, value_type>, bool> = true>
    constexpr default_value(const default_value<U> &other) : value_{ static_cast<value_type>(other.value_) }, use_default_init_{ other.use_default_init_ }, default_init_{ static_cast<value_type>(other.default_init_.value) } { }


    [[nodiscard]] constexpr const value_type &value() const noexcept {
        return use_default_init_ ? default_init_.value : value_;
    }
    [[nodiscard]] operator const value_type&() const noexcept{
        return this->value();
    }
    [[nodiscard]] constexpr const value_type &get_default() const noexcept {
        return default_init_.value;
    }
    [[nodiscard]] constexpr bool is_default() const noexcept {
        return use_default_init_;
    }

    constexpr void swap(default_value& other) noexcept(std::is_nothrow_swappable_v<T>) {
        using std::swap;
        swap(value_, other.value_);
        swap(use_default_init_, other.use_default_init_);
        swap(default_init_, other.default_init_);
    }
    constexpr void reset() noexcept {
        use_default_init_ = true;
    }

  private:
    value_type value_{};
    bool use_default_init_{ true };
    default_init<value_type> default_init_{};
};

template <typename T>
std::ostream& operator<<(std::ostream& out, const default_value<T> &val) {
    return out << val.value();
}
template <typename T>
std::istream &operator>>(std::istream &in, default_value<T> &val) {
    return in >> val.value_;
}

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


}


namespace std {

template <typename T>
struct hash<plssvm::default_value<T>> {
    std::size_t operator()(const plssvm::default_value<T> &val) const noexcept {
        return std::hash<T>{}(val.value());
    }
};

}