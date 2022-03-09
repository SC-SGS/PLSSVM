/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief MOCK class for the C-SVM class using the HIP backend.
 */

#pragma once

#include "plssvm/backends/HIP/csvm.hpp"                   // plssvm::hip::csvm
#include "plssvm/backends/HIP/detail/device_ptr.hip.hpp"  // plssvm::hip::detail::device_ptr
#include "plssvm/parameter.hpp"                           // plssvm::parameter

#include <vector>  // std::vector

/**
 * @brief GTest mock class for the HPI CSVM.
 * @tparam T the type of the data
 */
template <typename T>
class mock_hip_csvm : public plssvm::hip::csvm<T> {
    using base_type = plssvm::hip::csvm<T>;

  public:
    using real_type = typename base_type::real_type;
    using device_ptr_type = typename base_type::device_ptr_type;
    using queue_type = typename base_type::queue_type;

    explicit mock_hip_csvm(const plssvm::parameter<T> &params) :
        base_type{ params } {}

    // make non-virtual functions publicly visible
    using base_type::device_reduction;
    using base_type::generate_q;
    using base_type::run_device_kernel;
    using base_type::setup_data_on_device;

    // parameter setter
    void set_cost(const real_type cost) { base_type::cost_ = cost; }
    void set_QA_cost(const real_type QA_cost) { base_type::QA_cost_ = QA_cost; }

    // getter for internal variables
    std::shared_ptr<const std::vector<real_type>> &get_alpha_ptr() { return base_type::alpha_ptr_; }
    const std::vector<device_ptr_type> &get_device_data() const { return base_type::data_d_; }
    std::vector<queue_type> &get_devices() { return base_type::devices_; }
    typename std::vector<queue_type>::size_type get_num_devices() const { return base_type::devices_.size(); }
};