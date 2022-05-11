/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*/

#include "plssvm/detail/cmd/parameter_predict.hpp"

#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl_generic::list_available_sycl_implementations
#include "plssvm/constants.hpp"                          // plssvm::verbose
#include "plssvm/detail/string_utility.hpp"              // plssvm::detail::as_lower_case
#include "plssvm/parameter.hpp"                          // plssvm::parameter

#include "cxxopts.hpp"    // cxxopts::Options, cxxopts::value,cxxopts::ParseResult
#include "fmt/core.h"     // fmt::print, fmt::format
#include "fmt/ostream.h"  // can use fmt using operator<< overloads

#include <cstdio>      // stderr
#include <cstdlib>     // std::exit, EXIT_SUCCESS, EXIT_FAILURE
#include <exception>   // std::exception
#include <filesystem>  // std::filesystem::path

namespace plssvm::detail::cmd {

template <typename T>
parameter_predict<T>::parameter_predict(int argc, char **argv) {
   cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
   options
       .positional_help("test_file model_file [output_file]")
       .show_positional_help();
   options
       .set_width(150)
       .set_tab_expansion()
       // clang-format off
       .add_options()
           ("b,backend", fmt::format("choose the backend: {}", fmt::join(list_available_backends(), "|")), cxxopts::value<decltype(base_params.backend)>()->default_value(detail::as_lower_case(fmt::format("{}", base_params.backend))))
           ("p,target_platform", fmt::format("choose the target platform: {}", fmt::join(list_available_target_platforms(), "|")), cxxopts::value<decltype(base_params.target)>()->default_value(detail::as_lower_case(fmt::format("{}", base_params.target))))
#if defined(PLSSVM_HAS_SYCL_BACKEND)
           ("sycl_implementation_type", fmt::format("choose the SYCL implementation to be used in the SYCL backend: {}", fmt::join(sycl::list_available_sycl_implementations(), "|")), cxxopts::value<decltype(base_params.sycl_implementation_type)>()->default_value(detail::as_lower_case(fmt::format("{}", base_params.sycl_implementation_type))))
#endif
           ("use_strings_as_labels", "use strings as labels instead of plane numbers", cxxopts::value<bool>(base_params.strings_as_labels)->default_value(fmt::format("{}", base_params.strings_as_labels)))
           ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(plssvm::verbose)->default_value(fmt::format("{}", !plssvm::verbose)))
           ("h,help", "print this helper message", cxxopts::value<bool>())
           ("test", "", cxxopts::value<decltype(input_filename)>(), "test_file")
           ("model", "", cxxopts::value<decltype(model_filename)>(), "model_file")
           ("output", "", cxxopts::value<decltype(predict_filename)>(), "output_file");
   // clang-format on

   // parse command line options
   cxxopts::ParseResult result;
   try {
       options.parse_positional({ "test", "model", "output" });
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

   // parse backend_type and cast the value to the respective enum
   base_params.backend = result["backend"].as<decltype(base_params.backend)>();

   // parse target_platform and cast the value to the respective enum
   base_params.target = result["target_platform"].as<decltype(base_params.target)>();

#if defined(PLSSVM_HAS_SYCL_BACKEND)
   // parse SYCL implementation used in the SYCL backend
   base_params.sycl_implementation_type = result["sycl_implementation_type"].as<decltype(base_params.sycl_implementation_type)>();
#endif

   // parse whether strings should be used as labels
   base_params.strings_as_labels = result["use_strings_as_labels"].as<decltype(base_params.strings_as_labels)>();

   // parse whether output is quiet or not
   plssvm::verbose = !plssvm::verbose;

   // parse test data filename
   if (!result.count("test")) {
       fmt::print(stderr, "Error missing test file!");
       fmt::print("{}", options.help());
       std::exit(EXIT_FAILURE);
   }
   input_filename = result["test"].as<decltype(input_filename)>();

   // parse model filename
   if (!result.count("model")) {
       fmt::print(stderr, "Error missing model file!");
       fmt::print("{}", options.help());
       std::exit(EXIT_FAILURE);
   }
   model_filename = result["model"].as<decltype(model_filename)>();

   // parse output filename
   if (result.count("output")) {
       predict_filename = result["output"].as<decltype(predict_filename)>();
   } else {
       std::filesystem::path input_path{ input_filename };
       model_filename = input_path.replace_filename(input_path.filename().string() + ".predict").string();
   }
}

// explicitly instantiate template class
template class parameter_predict<float>;
template class parameter_predict<double>;


template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter_predict<T> &params) {
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
               "rho: {}\n"
               "input file (data set): '{}'\n"
               "input file (model): '{}'\n",
               "output file (prediction): '{}'\n",
               params.base_params.cost,
               params.base_params.epsilon,
               params.rho,
               params.input_filename,
               params.model_filename,
               params.predict_filename);
}
template std::ostream &operator<<(std::ostream &, const parameter_predict<float> &);
template std::ostream &operator<<(std::ostream &, const parameter_predict<double> &);

}  // namespace plssvm
