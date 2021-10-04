/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/backends/SYCL/detail/device_ptr.hpp"

#include "plssvm/backends/SYCL/exceptions.hpp"  // plssvm::sycl::backend_exception
#include "plssvm/detail/assert.hpp"             // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"     // sycl::detail::to_lower_case, sycl::detail::contains

#include "fmt/core.h"     // fmt::format
#include "sycl/sycl.hpp"  // sycl::queue, sycl::platform, sycl::gpu_selector

#include <cstddef>  // std::size_t
#include <utility>  // std::exchange, std::move, std::swap
#include <vector>   // std::vector

namespace plssvm::sycl::detail {

[[nodiscard]] std::vector<::sycl::queue> get_device_list_impl(const target_platform target) {
    std::vector<::sycl::queue> target_devices;
    for (const ::sycl::platform &platform : ::sycl::platform::get_platforms()) {
        for (const ::sycl::device &device : platform.get_devices()) {
            if (target == target_platform::cpu && device.is_cpu()) {
                // select CPU device
                target_devices.emplace_back(device, ::sycl::property::queue::in_order());
            } else {
                // must be a GPU device
                if (device.is_gpu()) {
                    // get vendor name of current GPU device
                    std::string vendor_string = device.get_info<::sycl::info::device::vendor>();
                    // convert vendor name to lower case
                    ::plssvm::detail::to_lower_case(vendor_string);
                    // get platform name of current GPU device
                    std::string platform_string = platform.get_info<::sycl::info::platform::name>();
                    // convert platform name to lower case
                    ::plssvm::detail::to_lower_case(platform_string);

                    switch (target) {
                        case target_platform::gpu_nvidia:
                            if (::plssvm::detail::contains(vendor_string, "nvidia")) {
                                target_devices.emplace_back(device, ::sycl::property::queue::in_order());
                            }
                            break;
                        case target_platform::gpu_amd:
                            if (::plssvm::detail::contains(vendor_string, "amd")) {
                                target_devices.emplace_back(device, ::sycl::property::queue::in_order());
                            }
                            break;
                        case target_platform::gpu_intel:
                            if (::plssvm::detail::contains(vendor_string, "intel")) {
#if PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_DPCPP
                                if (::plssvm::detail::contains(platform_string, PLSSVM_SYCL_BACKEND_DPCPP_BACKEND_TYPE)) {
                                    target_devices.emplace_back(device, ::sycl::property::queue::in_order());
                                }
#elif PLSSVM_SYCL_BACKEND_COMPILER == PLSSVM_SYCL_BACKEND_COMPILER_HIPSYCL
                                target_devices.emplace_back(device, ::sycl::property::queue::in_order());
#endif
                            }
                            break;
                        default:
                            break;
                    }
                }
            }
        }
    }
    return target_devices;
}

std::vector<::sycl::queue> get_device_list(const target_platform target) {
    if (target != target_platform::automatic) {
        return get_device_list_impl(target);
    } else {
        std::vector<::sycl::queue> target_devices = get_device_list_impl(target_platform::gpu_nvidia);
        if (target_devices.empty()) {
            target_devices = get_device_list_impl(target_platform::gpu_amd);
            if (target_devices.empty()) {
                target_devices = get_device_list_impl(target_platform::gpu_intel);
                if (target_devices.empty()) {
                    target_devices = get_device_list_impl(target_platform::cpu);
                }
            }
        }
        return target_devices;
    }
}

void device_synchronize(::sycl::queue &queue) {
    queue.wait_and_throw();
}

template <typename T>
device_ptr<T>::device_ptr(const size_type size, ::sycl::queue &queue) :
    queue_{ &queue }, size_{ size } {
    data_ = ::sycl::malloc_device<value_type>(size_, *queue_);
}

template <typename T>
device_ptr<T>::device_ptr(device_ptr &&other) noexcept :
    queue_{ std::exchange(other.queue_, nullptr) },
    data_{ std::exchange(other.data_, nullptr) },
    size_{ std::exchange(other.size_, 0) } {}

template <typename T>
device_ptr<T> &device_ptr<T>::operator=(device_ptr &&other) noexcept {
    device_ptr tmp{ std::move(other) };
    this->swap(tmp);
    return *this;
}

template <typename T>
device_ptr<T>::~device_ptr() {
    if (queue_ != nullptr) {
        ::sycl::free(static_cast<void *>(data_), *queue_);
    }
}

template <typename T>
void device_ptr<T>::swap(device_ptr &other) noexcept {
    std::swap(queue_, other.queue_);
    std::swap(data_, other.data_);
    std::swap(size_, other.size_);
}

template <typename T>
void device_ptr<T>::memset(const int value, const size_type pos) {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    this->memset(value, pos, size_);
}
template <typename T>
void device_ptr<T>::memset(const int value, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    if (pos >= size_) {
        throw backend_exception{ fmt::format("Illegal access in memset!: {} >= {}", pos, size_) };
    }
    const size_type rcount = std::min(count, size_ - pos);
    queue_->memset(static_cast<void *>(data_ + pos), value, rcount * sizeof(value_type)).wait();
}

template <typename T>
void device_ptr<T>::memcpy_to_device(const std::vector<value_type> &data_to_copy) {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    this->memcpy_to_device(data_to_copy, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_device(const std::vector<value_type> &data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    if (data_to_copy.size() < rcount) {
        throw backend_exception{ fmt::format("Too few data to perform memcpy (needed: {}, provided: {})!", rcount, data_to_copy.size()) };
    }
    this->memcpy_to_device(data_to_copy.data(), pos, rcount);
}
template <typename T>
void device_ptr<T>::memcpy_to_device(const_pointer data_to_copy) {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    this->memcpy_to_device(data_to_copy, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_device(const_pointer data_to_copy, const size_type pos, const size_type count) {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    queue_->copy(data_to_copy, data_, rcount).wait();
}

template <typename T>
void device_ptr<T>::memcpy_to_host(std::vector<value_type> &buffer) const {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    this->memcpy_to_host(buffer, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(std::vector<value_type> &buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    if (buffer.size() < rcount) {
        throw backend_exception{ fmt::format("Buffer too small to perform memcpy (needed: {}, provided: {})!", rcount, buffer.size()) };
    }
    this->memcpy_to_host(buffer.data(), pos, rcount);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(pointer buffer) const {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    this->memcpy_to_host(buffer, 0, size_);
}
template <typename T>
void device_ptr<T>::memcpy_to_host(pointer buffer, const size_type pos, const size_type count) const {
    PLSSVM_ASSERT(queue_ != nullptr, "Invalid sycl::queue!");
    PLSSVM_ASSERT(data_ != nullptr, "Invalid USM data pointer!");

    const size_type rcount = std::min(count, size_ - pos);
    queue_->copy(data_, buffer, rcount).wait();
}

template class device_ptr<float>;
template class device_ptr<double>;

}  // namespace plssvm::sycl::detail