// TODO:

#ifndef PLSSVM_DETAIL_SIMPLE_ANY
#define PLSSVM_DETAIL_SIMPLE_ANY
#pragma once

#include <memory>   // std::shared_ptr, std::make_shared, std::dynamic_pointer_cast
#include <utility>  // std::forward, std::move

namespace plssvm::detail {

struct simple_any {
    template <typename T>
    simple_any(T &&obj) :
        object{ std::make_shared<derived<T>>(std::forward<T>(obj)) } {}

    struct base {
        virtual ~base() = default;
    };

    template <typename T>
    struct derived : base {
        derived(const T &t) :
            obj{ t } {}
        derived(T &&t) :
            obj{ std::move(t) } {}

        T obj;
    };

    template <typename T>
    T &get() {
        return std::dynamic_pointer_cast<derived<T>>(object)->obj;
    }
    template <typename T>
    const T &get() const {
        return std::dynamic_pointer_cast<derived<T>>(object)->obj;
    }

    std::shared_ptr<base> object;
};

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_SIMPLE_ANY
