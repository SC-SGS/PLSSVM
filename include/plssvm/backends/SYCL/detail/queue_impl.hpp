//
// Created by breyerml on 02.12.22.
//

#ifndef PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_INCLUDE_PLSSVM_BACKENDS_SYCL_DETAIL_QUEUE_IMPL_HPP_
#define PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_INCLUDE_PLSSVM_BACKENDS_SYCL_DETAIL_QUEUE_IMPL_HPP_

#include "plssvm/backends/SYCL/detail/queue.hpp"

#include "sycl/sycl.hpp"

#include <utility>

namespace plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::detail {

struct queue::queue_impl {

    template <typename ... Args>
    queue_impl(Args... args) : sycl_queue{ std::forward<Args>(args)... } {}

    ::sycl::queue sycl_queue;
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_INCLUDE_PLSSVM_BACKENDS_SYCL_DETAIL_QUEUE_IMPL_HPP_
