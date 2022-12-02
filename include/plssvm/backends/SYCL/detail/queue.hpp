//
// Created by breyerml on 02.12.22.
//

#ifndef PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_INCLUDE_PLSSVM_BACKENDS_SYCL_DETAIL_QUEUE_HPP_
#define PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_INCLUDE_PLSSVM_BACKENDS_SYCL_DETAIL_QUEUE_HPP_

#include <memory>

namespace plssvm::@PLSSVM_SYCL_BACKEND_NAMESPACE_NAME@::detail {

class queue {
  public:
    struct queue_impl;
    std::shared_ptr<queue_impl> impl{};
};

}  // namespace plssvm::sycl::detail

#endif  // PLSSVM_PARALLEL_LEAST_SQUARES_SUPPORT_VECTOR_MACHINE_INCLUDE_PLSSVM_BACKENDS_SYCL_DETAIL_QUEUE_HPP_
