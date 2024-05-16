/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a `move_only_any` class based on [`std::any`](https://en.cppreference.com/w/cpp/utility/any) that works with move-only types.
 */

#ifndef PLSSVM_DETAIL_move_only_any_HPP_
#define PLSSVM_DETAIL_move_only_any_HPP_
#pragma once

#include "plssvm/detail/type_traits.hpp"  // PLSSVM_REQUIRES, plssvm::detail::remove_cvref_t

#include <initializer_list>  // std::initializer_list
#include <memory>            // std::unique_ptr, std::make_unique
#include <stdexcept>         // std::bad_cast
#include <type_traits>       // std::is_same_v, std::decay_t, std::is_constructible_v, std::is_void_v
#include <typeinfo>          // std::type_info
#include <utility>           // std::forward, std::move, std::in_place_type_t, std::in_place_type

namespace plssvm::detail {

/**
 * @brief The exception thrown in `plssvm::detail::move_only_any_cast` if the types mismatch.
 */
class bad_move_only_any_cast : public std::bad_cast {
  public:
    /**
     * @brief The exception's `what()` message.
     * @return the message string (`[[nodiscard]]`)
     */
    [[nodiscard]] const char *what() const noexcept final {
        return "plssvm::detail::bad_move_only_any_cast";
    }
};

/**
 * @brief A type erasure implementation.
 * @details The same idea as [`std::any`](https://en.cppreference.com/w/cpp/utility/any) which, however,
 *          can't be used since it requires the wrapped type to be copy constructible.
 *          Does **not** implement the encouraged optimizations for small objects.
 */
class move_only_any {
  private:
    // forward declare cast functions as friends
    template <typename T>
    friend T move_only_any_cast(const move_only_any &);
    template <typename T>
    friend T move_only_any_cast(move_only_any &);
    template <typename T>
    friend T move_only_any_cast(move_only_any &&);
    template <typename T>
    friend const T *move_only_any_cast(const move_only_any *) noexcept;
    template <typename T>
    friend T *move_only_any_cast(move_only_any *) noexcept;

    /**
     * @brief Type erase base class, such that `plssvm::detail::move_only_any` does not have to hold a templated member (which is not possible in C++).
     */
    struct type_erasure_base {
        type_erasure_base() = default;
        /**
         * @brief Default copy-constructor.
         */
        type_erasure_base(const type_erasure_base &) = default;
        /**
         * @brief Default move-constructor.
         */
        type_erasure_base(type_erasure_base &&) noexcept = default;
        /**
         * @brief Default copy-assignment operator.
         * @return `*this`
         */
        type_erasure_base &operator=(const type_erasure_base &) = default;
        /**
         * @brief Default move-assignment operator.
         * @return `*this`
         */
        type_erasure_base &operator=(type_erasure_base &&) noexcept = default;

        /**
         * @brief Virtual due to type_erasure_base being a polymorphic base class.
         */
        virtual ~type_erasure_base() = default;

        /**
         * @brief Pure virtual function to retrieve the type of the contained object.
         * @return the `typeid` of the contained object (`[[nodiscard]]`)
         */
        [[nodiscard]] virtual const std::type_info &type() const noexcept = 0;
    };

    /**
     * @brief The actual wrapper holding the type erased type.
     * @tparam T the type to hold
     */
    template <typename T>
    struct type_erasure_wrapper : type_erasure_base {
        /**
         * @brief Construct a type erased object using a forwarding reference.
         * @tparam Args the types of the parameter to construct the wrapped object with
         * @param[in,out] args the parameters to create the wrapped object
         */
        template <typename... Args>
        explicit type_erasure_wrapper(Args &&...args) :
            wrapped_object_(std::forward<Args>(args)...) { }  // cannot use curly braces here

        /**
         * @brief Obtain the type of the contained object.
         * @return the `typeid` of the contained object (`[[nodiscard]]`)
         */
        [[nodiscard]] const std::type_info &type() const noexcept override {
            return typeid(T);
        }

        /// The object stored in a type erased manner.
        T wrapped_object_;
    };

  public:
    /**
     * @brief The default construct. The `plssvm::detail::move_only_any` doesn't contain an entry.
     */
    move_only_any() noexcept = default;
    /**
     * @brief `plssvm::detail::move_only_any` does only support move-only types. Therefore, the copy constructor is deleted.
     */
    move_only_any(const move_only_any &) = delete;
    /**
     * @brief Default the move constructor.
     */
    move_only_any(move_only_any &&) noexcept = default;

    /**
     * @brief Construct a `plssvm::detail::move_only_any` holding the object @p value.
     * @tparam ValueType the contained object will have the type `std::decay_t<ValueType>`
     * @param[in,out] value the object to type erase
     */
    template <typename ValueType,
              PLSSVM_REQUIRES(!std::is_same_v<std::decay_t<ValueType>, move_only_any>)>
    explicit move_only_any(ValueType &&value) :
        object_ptr_{ std::make_unique<type_erasure_wrapper<std::decay_t<ValueType>>>(std::forward<ValueType>(value)) } { }

    /**
     * @brief Construct a `plssvm::detail::move_only_any` holding an object of type @p ValueType inplace using @p args.
     * @tparam ValueType the contained object will have the type `std::decay_t<ValueType>`
     * @tparam Args the types used to construct @p ValueType
     * @param[in,out] args the parameters used to construct an object of type @p ValueType inplace
     */
    template <typename ValueType, typename... Args, PLSSVM_REQUIRES(std::is_constructible_v<std::decay_t<ValueType>, Args...>)>
    explicit move_only_any(std::in_place_type_t<ValueType>, Args &&...args) :
        object_ptr_{ std::make_unique<type_erasure_wrapper<std::decay_t<ValueType>>>(std::forward<Args>(args)...) } { }

    /**
     * @brief Construct a `plssvm::detail::move_only_any` holding an object of type @p ValueType inplace using @p il and @p args.
     * @tparam ValueType the contained object will have the type `std::decay_t<ValueType>`
     * @tparam U the types in the `std::initializer_list`
     * @tparam Args the types used to construct @p ValueType
     * @param[in] il the `std::initializer_list` to construct an object of type @p ValueType
     * @param[in,out] args the parameters used to construct an object of type @p ValueType inplace
     */
    template <typename ValueType, typename U, typename... Args, PLSSVM_REQUIRES(std::is_constructible_v<std::decay_t<ValueType>, std::initializer_list<U>, Args...>)>
    explicit move_only_any(std::in_place_type_t<ValueType>, std::initializer_list<U> il, Args &&...args) :
        object_ptr_{ std::make_unique<type_erasure_wrapper<std::decay_t<ValueType>>>(il, std::forward<Args>(args)...) } { }

    /**
     * @brief `plssvm::detail::move_only_any` does only support move-only types. Therefore, the copy-assignment operator is deleted.
     */
    move_only_any &operator=(const move_only_any &) = delete;
    /**
     * @brief Default the move-assignment operator.
     * @return `*this`
     */
    move_only_any &operator=(move_only_any &&) noexcept = default;

    /**
     * @brief Assign the object @p rhs to the `plssvm::detail::move_only_any`.
     * @tparam ValueType the contained object will have the type `std::decay_t<ValueType>`
     * @param[in,out] rhs the object to type erase
     * @return `*this`
     */
    template <typename ValueType, PLSSVM_REQUIRES(!std::is_same_v<std::decay_t<ValueType>, move_only_any>)>
    move_only_any &operator=(ValueType &&rhs) {
        move_only_any(std::forward<ValueType>(rhs)).swap(*this);
        return *this;
    }

    /**
     * @brief Default the destructor.
     */
    ~move_only_any() = default;

    /**
     * @brief Inplace construct, and possibly override, the object contained in this `plssvm::detail::move_only_any` using @p args.
     * @tparam ValueType the contained object will have the type `std::decay_t<ValueType>`
     * @tparam Args the types used to construct @p ValueType
     * @param[in,out] args the parameters used to construct an object of type @p ValueType inplace
     * @return a reference to the newly constructed object
     */
    template <typename ValueType, typename... Args, PLSSVM_REQUIRES(std::is_constructible_v<std::decay_t<ValueType>, Args...>)>
    std::decay_t<ValueType> &emplace(Args &&...args) {
        object_ptr_ = std::make_unique<type_erasure_wrapper<std::decay_t<ValueType>>>(std::forward<Args>(args)...);
        return dynamic_cast<move_only_any::type_erasure_wrapper<ValueType> &>(*object_ptr_).wrapped_object_;
    }

    /**
     * @brief Inplace construct, and possibly override, the object contained in this `plssvm::detail::move_only_any` using @p il and @p args.
     * @tparam ValueType the contained object will have the type `std::decay_t<ValueType>`
     * @tparam U the types in the `std::initializer_list`
     * @tparam Args the types used to construct @p ValueType
     * @param[in] il the `std::initializer_list` to construct an object of type @p ValueType
     * @param[in] args the values used to construct @p ValueType
     * @return a reference to the newly constructed object
     */
    template <typename ValueType, typename U, typename... Args, PLSSVM_REQUIRES(std::is_constructible_v<std::decay_t<ValueType>, std::initializer_list<U>, Args...>)>
    std::decay_t<ValueType> &emplace(std::initializer_list<U> il, Args &&...args) {
        object_ptr_ = std::make_unique<type_erasure_wrapper<std::decay_t<ValueType>>>(il, std::forward<Args>(args)...);
        return dynamic_cast<move_only_any::type_erasure_wrapper<ValueType> &>(*object_ptr_).wrapped_object_;
    }

    /**
     * @brief If not empty, destroys the contained object.
     */
    void reset() noexcept {
        object_ptr_.reset(nullptr);
    }

    /**
     * @brief Swaps the content of two `plssvm::detail::move_only_any` objects.
     * @param[in,out] other the other move_only_any
     */
    void swap(move_only_any &other) noexcept {
        object_ptr_.swap(other.object_ptr_);
    }

    /**
     * @brief Checks whether the object contains a value.
     * @return `true` if the `plssvm::detail::move_only_any` isn't empty, otherwise `false` (`[[nodiscard]]`)
     */
    [[nodiscard]] bool has_value() const noexcept {
        return object_ptr_ != nullptr;
    }

    /**
     * @brief Queries the contained type.
     * @return The typeid of the contained value if instance is non-empty, otherwise `typeid(void)`.
     */
    [[nodiscard]] const std::type_info &type() const noexcept {
        if (this->has_value()) {
            // return the type if this isn't empty
            return object_ptr_->type();
        }
        // return value according to the std::any specification
        return typeid(void);
    }

  private:
    /// The type erased object to hold.
    std::unique_ptr<type_erasure_base> object_ptr_{ nullptr };
};

/**
 * @brief Swaps the content of two `plssvm::detail::move_only_any` objects.
 * @details Calls: `lhs.swap(rhs)`.
 * @param[in,out] lhs the first any object
 * @param[in,out] rhs the second any object
 */
inline void swap(move_only_any &lhs, move_only_any &rhs) noexcept {
    lhs.swap(rhs);
}

/**
 * @brief Performs type-safe access to the contained object in @p operand.
 * @tparam T the type of the contained object
 * @param[in] operand the `plssvm::detail::move_only_any` containing the object
 * @return the contained object (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T move_only_any_cast(const move_only_any &operand) {
    using U = detail::remove_cvref_t<T>;
    static_assert(std::is_constructible_v<T, const U &>);
    if (auto *ptr = move_only_any_cast<U>(&operand)) {
        // get the value if possible
        return static_cast<T>(*ptr);
    }
    // if a nullptr is returned (dynamic_cast failed) throw a bad_move_only_cast exception
    throw bad_move_only_any_cast{};
}

/**
 * @brief Performs type-safe access to the contained object in @p operand.
 * @tparam T the type of the contained object
 * @param[in] operand the `plssvm::detail::move_only_any` containing the object
 * @return the contained object (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T move_only_any_cast(move_only_any &operand) {
    using U = detail::remove_cvref_t<T>;
    static_assert(std::is_constructible_v<T, U &>);
    if (auto *ptr = move_only_any_cast<U>(&operand)) {
        // get the value if possible
        return static_cast<T>(*ptr);
    }
    // if a nullptr is returned (dynamic_cast failed) throw a bad_move_only_cast exception
    throw bad_move_only_any_cast{};
}

/**
 * @brief Performs type-safe access to the contained object in @p operand.
 * @tparam T the type of the contained object
 * @param[in] operand the `plssvm::detail::move_only_any` containing the object
 * @return the contained object (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T move_only_any_cast(move_only_any &&operand) {
    using U = detail::remove_cvref_t<T>;
    static_assert(std::is_constructible_v<T, U>);
    if (auto *ptr = move_only_any_cast<U>(&operand)) {
        // get the value if possible
        return static_cast<T>(std::move(*ptr));
    }
    // if a nullptr is returned (dynamic_cast failed) throw a bad_move_only_cast exception
    throw bad_move_only_any_cast{};
}

/**
 * @brief Performs type-safe access to the contained object in @p operand.
 * @tparam T the type of the contained object
 * @param[in] operand the `plssvm::detail::move_only_any` containing the object
 * @return the contained object (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline const T *move_only_any_cast(const move_only_any *operand) noexcept {
    using U = detail::remove_cvref_t<T>;
    static_assert(!std::is_void_v<T>);
    if (operand == nullptr) {
        // return a nullptr if a nullptr is provided
        return nullptr;
    } else if (auto *ptr = dynamic_cast<const move_only_any::type_erasure_wrapper<U> *>(std::addressof(*operand->object_ptr_))) {
        return std::addressof(ptr->wrapped_object_);
    }
    return nullptr;
}

/**
 * @brief Performs type-safe access to the contained object in @p operand.
 * @tparam T the type of the contained object
 * @param[in] operand the `plssvm::detail::move_only_any` containing the object
 * @return the contained object (`[[nodiscard]]`)
 */
template <typename T>
[[nodiscard]] inline T *move_only_any_cast(move_only_any *operand) noexcept {
    using U = detail::remove_cvref_t<T>;
    static_assert(!std::is_void_v<T>);
    if (operand == nullptr) {
        // return a nullptr if a nullptr is provided
        return nullptr;
    } else if (auto *ptr = dynamic_cast<move_only_any::type_erasure_wrapper<U> *>(std::addressof(*operand->object_ptr_))) {
        return std::addressof(ptr->wrapped_object_);
    }
    return nullptr;
}

/**
 * @brief Constructs a `plssvm::detail::move_only_any` object containing an object of type @p T, passing the provided arguments @p args to @p T's constructor.
 * @tparam T the type of the contained object
 * @tparam Args the types of the parameters used to construct the object of type @p T
 * @param[in,out] args the parameters to construct the contained object
 * @return the newly constructed `plssvm::detail::move_only_any` (`[[nodiscard]]`)
 */
template <typename T, typename... Args>
[[nodiscard]] inline move_only_any make_move_only_any(Args &&...args) {
    return move_only_any(std::in_place_type<T>, std::forward<Args>(args)...);
}

/**
 * @brief Constructs a `plssvm::detail::move_only_any` object containing an object of type @p T, passing the provided arguments @p il and @p args to @p T's constructor.
 * @tparam T the type of the contained object
 * @tparam U the types in the `std::initializer_list`
 * @tparam Args the types of the parameters used to construct the object of type @p T
 * @param[in] il the `std::initializer_list` to construct an object of type @p ValueType
 * @param[in,out] args the parameters to construct the contained object
 * @return the newly constructed `plssvm::detail::move_only_any` (`[[nodiscard]]`)
 */
template <typename T, typename U, typename... Args>
[[nodiscard]] inline move_only_any make_move_only_any(std::initializer_list<U> il, Args &&...args) {
    return move_only_any(std::in_place_type<T>, il, std::forward<Args>(args)...);
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_move_only_any_HPP_
