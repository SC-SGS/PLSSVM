/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines very simple type erasure class.
 */

#ifndef PLSSVM_DETAIL_SIMPLE_ANY_HPP_
#define PLSSVM_DETAIL_SIMPLE_ANY_HPP_
#pragma once

#include <memory>   // std::unique_ptr, std::make_unique
#include <utility>  // std::forward

namespace plssvm::detail {

class simple_any {
  private:
    /**
     * @brief Type erase base class, such that `simple_any` does not have to hold a templated member (which is not possible in C++).
     */
    struct type_erasure_base {
        /**
         * @brief Virtual due to type_erasure_base being a polymorphic base class.
         */
        virtual ~type_erasure_base() = default;
    };

    /**
     * @brief The actual wrapper holding the type erased type.
     * @tparam T the type to hold
     */
    template <typename T>
    struct type_erasure_wrapper : type_erasure_base {
        /**
         * @brief Construct type erased object using a forwarding reference.
         * @param[in,out] obj the object to create the wrapped object
         */
        template <typename U>
        explicit type_erasure_wrapper(U &&obj) :
            wrapped_object_{ std::forward<U>(obj) } {}

        /// The object stored in a type erased manner.
        T wrapped_object_;
    };

  public:
    /**
     * @brief Construct the type erased object by forwarding the construction to the actual type erasing wrapper.
     * @tparam T the type of the type erased object
     * @param[in,out] obj the object to type erase
     */
    template <typename T>
    explicit simple_any(T &&obj) :
        object_ptr_{ std::make_unique<type_erasure_wrapper<T>>(std::forward<T>(obj)) } {}

    /**
     * @brief Access the type erased object. Returns `nullptr` if the type erased object isn not of type `T`.
     * @tparam T the expected type of the type erased object
     * @return the object (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] T &get() {
        return dynamic_cast<type_erasure_wrapper<T> &>(*object_ptr_).wrapped_object_;
    }
    /**
     * @brief Access the type erased object. Returns `nullptr` if the type erased object isn not of type `T`.
     * @tparam T the expected type of the type erased object
     * @return the object (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] const T &get() const {
        return dynamic_cast<const type_erasure_wrapper<T> &>(*object_ptr_).wrapped_object_;
    }

    /**
     * @brief Access the type erased object and move it out of the simple_any wrapper.
     * @tparam T the expected type of the type erased object
     * @return the object as rvalue (`[[nodiscard]]`)
     */
    template <typename T>
    [[nodiscard]] T &&move() {
        return std::move(dynamic_cast<type_erasure_wrapper<T> &&>(*object_ptr_).wrapped_object_);
    }

  private:
    /// The type erased object to hold.
    std::unique_ptr<type_erasure_base> object_ptr_;
};

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_SIMPLE_ANY_HPP_
