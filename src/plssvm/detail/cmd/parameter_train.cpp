/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*/

#include "plssvm/detail/cmd/parameter_train.hpp"

#include "plssvm/backend_types.hpp"                      // plssvm::list_available_backends
#include "plssvm/kernel_types.hpp"                       // plssvm::kernel_type_to_math_string
#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl_generic::list_available_sycl_implementations
#include "plssvm/constants.hpp"                          // plssvm::verbose
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::as_lower_case
#include "plssvm/detail/utility.hpp"                     // plssvm::detail::to_underlying
#include "plssvm/target_platforms.hpp"                   // plssvm::list_available_target_platforms

#include "cxxopts.hpp"    // cxxopts::Options, cxxopts::value,cxxopts::ParseResult
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstdio>      // stderr
#include <cstdlib>     // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>   // std::exception
#include <filesystem>  // std::filesystem::path

namespace plssvm::detail::cmd {

template <typename T>
parameter_train<T>::parameter_train(int argc, char **argv) {
   cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
   options
       .positional_help("training_set_file [model_file]")
       .show_positional_help();
   options
       .set_width(150)
       .set_tab_expansion()
       // clang-format off
       .add_options()
           ("t,kernel_type", "set type of kernel function. \n\t 0 -- linear: u'*v\n\t 1 -- polynomial: (gamma*u'*v + coef0)^degree \n\t 2 -- radial basis function: exp(-gamma*|u-v|^2)", cxxopts::value<decltype(base_params.kernel)>()->default_value(fmt::format("{}", detail::to_underlying(base_params.kernel))))
           ("d,degree", "set degree in kernel function", cxxopts::value<decltype(base_params.degree)>()->default_value(fmt::format("{}", base_params.degree)))
           ("g,gamma", "set gamma in kernel function (default: 1 / num_features)", cxxopts::value<decltype(base_params.gamma)>())
           ("r,coef0", "set coef0 in kernel function", cxxopts::value<decltype(base_params.coef0)>()->default_value(fmt::format("{}", base_params.coef0)))
           ("c,cost", "set the parameter C", cxxopts::value<decltype(base_params.cost)>()->default_value(fmt::format("{}", base_params.cost)))
           ("e,epsilon", "set the tolerance of termination criterion", cxxopts::value<decltype(base_params.epsilon)>()->default_value(fmt::format("{}", base_params.epsilon)))
           ("b,backend", fmt::format("choose the backend: {}", fmt::join(list_available_backends(), "|")), cxxopts::value<decltype(base_params.backend)>()->default_value(detail::as_lower_case(fmt::format("{}", base_params.backend))))
           ("p,target_platform", fmt::format("choose the target platform: {}", fmt::join(list_available_target_platforms(), "|")), cxxopts::value<decltype(base_params.target)>()->default_value(detail::as_lower_case(fmt::format("{}", base_params.target))))
#if defined(PLSSVM_HAS_SYCL_BACKEND)
           ("sycl_kernel_invocation_type", "choose the kernel invocation type when using SYCL as backend: automatic|nd_range|hierarchical", cxxopts::value<decltype(base_params.sycl_kernel_invocation_type)>()->default_value(detail::as_lower_case(fmt::format("{}", base_params.sycl_kernel_invocation_type))))
           ("sycl_implementation_type", fmt::format("choose the SYCL implementation to be used in the SYCL backend: {}", fmt::join(sycl::list_available_sycl_implementations(), "|")), cxxopts::value<decltype(base_params.sycl_implementation_type)>()->default_value(detail::as_lower_case(fmt::format("{}", base_params.sycl_implementation_type))))
#endif
           ("use_strings_as_labels", "use strings as labels instead of plane numbers", cxxopts::value<bool>(base_params.strings_as_labels)->default_value(fmt::format("{}", base_params.strings_as_labels)))
           ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(plssvm::verbose)->default_value(fmt::format("{}", !plssvm::verbose)))
           ("h,help", "print this helper message", cxxopts::value<bool>())
           ("input", "", cxxopts::value<decltype(input_filename)>(), "training_set_file")
           ("model", "", cxxopts::value<decltype(model_filename)>(), "model_file");
   // clang-format on

   // parse command line options
   cxxopts::ParseResult result;
   try {
       options.parse_positional({ "input", "model" });
       result = options.parse(argc, argv);
   } catch (const std::exception &e) {
       fmt::print("{}\n{}\n", e.what(), options.help());
       std::exit(EXIT_FAILURE);
   }

   // print help message and exit
   if (result.count("help")) {
       fmt::print("{}", options.help());
       std::exit(EXIT_SUCCESS);
   }

   // parse kernel_type and cast the value to the respective enum
   base_params.kernel = result["kernel_type"].as<decltype(base_params.kernel)>();

   // parse degree
   base_params.degree = result["degree"].as<decltype(base_params.degree)>();

   // parse gamma
   if (result.count("gamma")) {
       base_params.gamma = result["gamma"].as<decltype(base_params.gamma)>();
       if (base_params.gamma == decltype(base_params.gamma){ 0.0 }) {
           fmt::print(stderr, "gamma = 0.0 is not allowed, it doesnt make any sense!\n");
           fmt::print("{}", options.help());
           std::exit(EXIT_FAILURE);
       }
   } else {
       base_params.gamma = decltype(base_params.gamma){ 0.0 };
   }

   // parse coef0
   base_params.coef0 = result["coef0"].as<decltype(base_params.coef0)>();

   // parse cost
   base_params.cost = result["cost"].as<decltype(base_params.cost)>();

   // parse epsilon
   base_params.epsilon = result["epsilon"].as<decltype(base_params.epsilon)>();

   // parse backend_type and cast the value to the respective enum
   base_params.backend = result["backend"].as<decltype(base_params.backend)>();

   // parse target_platform and cast the value to the respective enum
   base_params.target = result["target_platform"].as<decltype(base_params.target)>();

#if defined(PLSSVM_HAS_SYCL_IMPLEMENTATION)
   // parse kernel invocation type when using SYCL as backend
   base_params.sycl_kernel_invocation_type = result["sycl_kernel_invocation_type"].as<decltype(base_params.sycl_kernel_invocation_type)>();

   // parse SYCL implementation used in the SYCL backend
   base_params.sycl_implementation_type = result["sycl_implementation_type"].as<decltype(base_params.sycl_implementation_type)>();
#endif

   // parse whether strings should be used as labels
   base_params.strings_as_labels = result["use_strings_as_labels"].as<decltype(base_params.strings_as_labels)>();

   // parse whether output is quiet or not
   plssvm::verbose = !plssvm::verbose;

   // parse input data filename
   if (!result.count("input")) {
       fmt::print(stderr, "Error missing input file!");
       fmt::print("{}", options.help());
       std::exit(EXIT_FAILURE);
   }
   input_filename = result["input"].as<decltype(input_filename)>();

   // parse output model filename
   if (result.count("model")) {
       model_filename = result["model"].as<decltype(model_filename)>();
   } else {
       std::filesystem::path input_path{ input_filename };
       model_filename = input_path.replace_filename(input_path.filename().string() + ".model").string();
   }
}

// explicitly instantiate template class
template class parameter_train<float>;
template class parameter_train<double>;


template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter_train<T> &params) {
    out << fmt::format("kernel_type: {} -> {}\n", params.base_params.kernel, kernel_type_to_math_string(params.base_params.kernel));
    switch (params.base_params.kernel) {
        case kernel_type::linear:
            break;
        case kernel_type::polynomial:
            out << fmt::format(
                "gamma: {}\n"
                "coef0: {}\n"
                "degree: {}\n",
                params.base_params.gamma,
                params.base_params.coef0,
                params.base_params.degree);
            break;
        case kernel_type::rbf:
            out << fmt::format("gamma: {}\n", params.base_params.gamma);
            break;
    }
    return out << fmt::format(
               "cost: {}\n"
               "epsilon: {}\n"
               "use strings as labels: {}\n"
               "input file (data set): '{}'\n"
               "output file (model): '{}'\n",
               params.base_params.cost,
               params.base_params.epsilon,
               params.input_filename,
               params.model_filename);
}
template std::ostream &operator<<(std::ostream &, const parameter_train<float> &);
template std::ostream &operator<<(std::ostream &, const parameter_train<double> &);

}  // namespace plssvm