/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines a struct containing the GPU backend's execution ranges.
 */

#ifndef PLSSVM_BACKENDS_EXECUTION_RANGE_HPP_
#define PLSSVM_BACKENDS_EXECUTION_RANGE_HPP_

#include "fmt/base.h"     // fmt::formatter
#include "fmt/ostream.h"  // fmt::ostream_formatter

#include <iosfwd>   // forward declare std::ostream and std::istream
#include <utility>  // std::pair
#include <vector>   // std::vector

namespace plssvm::detail {

//*************************************************************************************************************************************//
//                                                              dim_type                                                               //
//*************************************************************************************************************************************//

/**
 * @brief A type encapsulating up-to three dimensions for kernel launches.
 */
struct [[nodiscard]] dim_type {
    /**
     * @brief Construct an empty dimensional type.
     */
    constexpr dim_type() = default;

    /**
     * @brief Construct an one-dimensional dimensional type.
     * @param[in] x_p the value of the first dimension
     */
    constexpr explicit dim_type(const unsigned long long x_p) :
        x{ x_p } { }

    /**
     * @brief Construct a two-dimensional dimensional type.
     * @param[in] x_p the value of the first dimension
     * @param[in] y_p the value of the second dimension
     */
    constexpr dim_type(const unsigned long long x_p, const unsigned long long y_p) :
        x{ x_p },
        y{ y_p } { }

    /**
     * @brief Construct a three-dimensional dimensional type.
     * @param[in] x_p the value of the first dimension
     * @param[in] y_p the value of the second dimension
     * @param[in] z_p the value of the third dimension
     */
    constexpr dim_type(const unsigned long long x_p, const unsigned long long y_p, const unsigned long long z_p) :
        x{ x_p },
        y{ y_p },
        z{ z_p } { }

    /**
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[in,out] other the other dim_type
     */
    constexpr void swap(dim_type &other) noexcept {
        // can't use std::swap since it isn't constexpr in C++17
        constexpr auto swap_ull = [](auto &lhs, auto &rhs) {
            auto temp{ rhs };
            rhs = lhs;
            lhs = temp;
        };
        swap_ull(x, other.x);
        swap_ull(y, other.y);
        swap_ull(z, other.z);
    }

    /**
     * @brief Return the total number of elements in the dimensional type.
     * @details Equal to: `x * y * z`.
     * @return the total number of elements (`[[nodiscard]]`)
     */
    [[nodiscard]] constexpr unsigned long long total_size() const noexcept {
        return x * y * z;
    }

    /// The dimensional size in x direction.
    unsigned long long x{ 1 };
    /// The dimensional size in y direction.
    unsigned long long y{ 1 };
    /// The dimensional size in z direction.
    unsigned long long z{ 1 };
};

/**
 * @brief Swap the contents of @p lhs with the contents of @p rhs.
 * @param[in,out] lhs the first dim_type
 * @param[in,out] rhs the second dim_type
 */
constexpr void swap(dim_type &lhs, dim_type &rhs) noexcept {
    lhs.swap(rhs);
}

/**
 * @brief Compare two dim_types for equality.
 * @param[in] lhs the first dim_type
 * @param[in] rhs the second dim_type
 * @return `true` if all three dimensions are equal, otherwise `false` (`[[nodiscard]]`)
 */
constexpr bool operator==(const dim_type lhs, const dim_type rhs) {
    return lhs.x == rhs.x && lhs.y == rhs.y && lhs.z == rhs.z;
}

/**
 * @brief Compare two dim_types for inequality.
 * @param[in] lhs the first dim_type
 * @param[in] rhs the second dim_type
 * @return `false` if all three dimensions are equal, otherwise `true` (`[[nodiscard]]`)
 */
constexpr bool operator!=(const dim_type lhs, const dim_type rhs) {
    return !(lhs == rhs);
}

/**
 * @brief Output the @p dim to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the dim_type to
 * @param[in] dim the dim_type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, dim_type dim);

//*************************************************************************************************************************************//
//                                                           execution_range                                                           //
//*************************************************************************************************************************************//

/**
 * @brief A struct encapsulating an arbitrary execution range used to launch a kernel.
 */
struct execution_range {
    /// The type used to store the grid sizes (first) and offsets (second).
    using grid_type = std::pair<dim_type, dim_type>;

    /**
     * @brief Create a block and grid(s) used to launch the kernels.
     * @details If the provided grid would be too large to be launched, it is split into multiple sub-grids.
     * @param[in] block_p the requested block size
     * @param[in] max_allowed_block_size the maximum allowed 1D block size (i.e., `block.x * block.y * block.z`)
     * @param[in] grid the requested grid size
     * @param[in] max_allowed_grid_size the maximum allowed 3D grid sizes
     * @throws plssvm::kernel_launch_resources if the block size exceeds the upper limits
     */
    execution_range(dim_type block_p, unsigned long long max_allowed_block_size, dim_type grid, dim_type max_allowed_grid_size);

    /**
     * @brief Swap the contents of `*this` with the contents of @p other.
     * @param[in,out] other the other execution range
     */
    void swap(execution_range &other) noexcept;

    /**
     * @brief Calculate the number of threads in a block described by this execution range.
     * @return the number of threads, i.e., `block.x * block.y * block.z` (`[[nodiscard]]`)
     */
    [[nodiscard]] unsigned long long num_threads_in_block() const noexcept;

    /// The up-to three dimensional block (work-group) size.
    dim_type block{};
    /// The grids. Multiple grids are used, if the grid sizes would exceed the maximum allowed number. Also stores the offsets for the respective grids used in the kernels.
    /// Note: no default initialization due to a linker error occurring with NVIDIA's nvhpc!
    std::vector<grid_type> grids;

};

/**
 * @brief Swap the contents of @p lhs with the contents of @p rhs.
 * @param[in,out] lhs the first execution range
 * @param[in,out] rhs the second execution range
 */
void swap(execution_range &lhs, execution_range &rhs) noexcept;

/**
 * @brief Compare two execution ranges for equality.
 * @param[in] lhs the first execution range
 * @param[in] rhs the second execution range
 * @return `true` if all grids and blocks are equal, otherwise `false` (`[[nodiscard]]`)
 */
constexpr bool operator==(const execution_range &lhs, const execution_range &rhs) {
    return lhs.block == rhs.block && lhs.grids == rhs.grids;
}

/**
 * @brief Compare two execution ranges for inequality.
 * @param[in] lhs the first execution range
 * @param[in] rhs the second execution range
 * @return `false` if all grids and blocks are equal, otherwise `true` (`[[nodiscard]]`)
 */
constexpr bool operator!=(const execution_range &lhs, const execution_range &rhs) {
    return !(lhs == rhs);
}

/**
 * @brief Output the execution_range @p exec to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the execution range to
 * @param[in] exec the execution range
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, const execution_range &exec);

}  // namespace plssvm::detail

/// @cond Doxygen_suppress

template <>
struct fmt::formatter<plssvm::detail::dim_type> : fmt::ostream_formatter { };

template <>
struct fmt::formatter<plssvm::detail::execution_range> : fmt::ostream_formatter { };

/// @endcond

#endif  // PLSSVM_BACKENDS_EXECUTION_RANGE_HPP_
