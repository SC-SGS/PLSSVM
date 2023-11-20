/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Utility functions and aliases related to the igor library.
 */

#ifndef PLSSVM_DETAIL_IGOR_UTILITY_HPP_
#define PLSSVM_DETAIL_IGOR_UTILITY_HPP_

#include "plssvm/detail/type_traits.hpp"  // plssvm::detail::{remove_cvref_t, always_false_v}
#include "plssvm/detail/utility.hpp"      // plssvm::detail::unreachable

#include "igor/igor.hpp"  // igor::parser, igor::has_unnamed_arguments

namespace plssvm::detail {

/**
 * @brief Trait to check whether @p Args only contains named-parameter.
 */
template <typename... Args>
constexpr bool has_only_named_args_v = !igor::has_unnamed_arguments<Args...>();

/**
 * @brief Parse the value hold be @p named_arg and return it converted to the @p ExpectedType.
 * @tparam ExpectedType the type the value of the named argument should be converted to
 * @tparam IgorParser the type of the named-parameter parser
 * @tparam NamedArgType the type of the named-parameter (necessary since they are struct tags)
 * @param[in] parser the named-parameter parser
 * @param[in] named_arg the named-parameter argument
 * @return the value of @p named_arg converted to @p ExpectedType (`[[nodiscard]]`)
 */
template <typename ExpectedType, typename IgorParser, typename NamedArgType>
ExpectedType get_value_from_named_parameter(const IgorParser &parser, const NamedArgType &named_arg) {
    using parsed_named_arg_type = detail::remove_cvref_t<decltype(parser(named_arg))>;
    // check whether a plssvm::default_value (e.g., plssvm::default_value<double>) or unwrapped normal value (e.g., double) has been provided
    if constexpr (is_default_value_v<parsed_named_arg_type>) {
        static_assert(std::is_convertible_v<typename parsed_named_arg_type::value_type, ExpectedType>, "Cannot convert the wrapped default value to the expected type!");
        // a plssvm::default_value has been provided (e.g., plssvm::default_value<double>)
        return static_cast<ExpectedType>(parser(named_arg).value());
    } else if constexpr (std::is_convertible_v<parsed_named_arg_type, ExpectedType>) {
        // an unwrapped value has been provided (e.g., double)
        return static_cast<ExpectedType>(parser(named_arg));
    } else {
        static_assert(detail::always_false_v<ExpectedType>, "Cannot convert the named argument to the ExpectedType or plssvm::default_value<ExpectedType>!");
    }
    // may never be reached
    detail::unreachable();
}

}  // namespace plssvm::detail

#endif  // PLSSVM_DETAIL_IGOR_UTILITY_HPP_
