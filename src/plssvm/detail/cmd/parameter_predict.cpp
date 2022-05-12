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

parameter_predict::parameter_predict(int argc, char **argv) {
   cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
   options
       .positional_help("test_file model_file [output_file]")
       .show_positional_help();
   options
       .set_width(150)
       .set_tab_expansion()
       // clang-format off
       .add_options()
           ("b,backend", fmt::format("choose the backend: {}", fmt::join(list_available_backends(), "|")), cxxopts::value<backend_type>()->default_value(detail::as_lower_case("automatic")))
           ("p,target_platform", fmt::format("choose the target platform: {}", fmt::join(list_available_target_platforms(), "|")), cxxopts::value<target_platform>()->default_value(detail::as_lower_case("automatic")))
#if defined(PLSSVM_HAS_SYCL_BACKEND)
           ("sycl_implementation_type", fmt::format("choose the SYCL implementation to be used in the SYCL backend: {}", fmt::join(sycl::list_available_sycl_implementations(), "|")), cxxopts::value<sycl::implementation_type>()->default_value(detail::as_lower_case("automatic")))
#endif
           ("use_strings_as_labels", "use strings as labels instead of plane numbers", cxxopts::value<bool>()->default_value("false"))
           ("use_float_as_real_type", "use floats as real types instead of doubles", cxxopts::value<bool>()->default_value("false"))
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

   // instantiate variant
   if (result["use_float_as_real_type"].as<bool>()) {
       base_params = parameter_variants{ parameter<float>{} };
   } else {
       base_params = parameter_variants { parameter<double>{} };
   }

   // parse base_params values
   std::visit([&](auto&& params) {
       // parse backend_type and cast the value to the respective enum
       params.backend = result["backend"].as<decltype(params.backend)>();

       // parse target_platform and cast the value to the respective enum
       params.target = result["target_platform"].as<decltype(params.target)>();

#if defined(PLSSVM_HAS_SYCL_BACKEND)
       // parse SYCL implementation used in the SYCL backend
       params.sycl_implementation_type = result["sycl_implementation_type"].as<decltype(params.sycl_implementation_type)>();
#endif

       // parse whether strings should be used as labels
       params.strings_as_labels = result["use_strings_as_labels"].as<decltype(params.strings_as_labels)>();
   }, base_params);

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

std::ostream &operator<<(std::ostream &out, const parameter_predict &params) {
    std::visit([&](auto&& args) {
        out << fmt::format("kernel_type: {} -> {}\n", args.kernel, kernel_type_to_math_string(args.kernel));
        switch (args.kernel) {
            case kernel_type::linear:
                break;
            case kernel_type::polynomial:
                out << fmt::format(
                    "gamma: {}\n"
                    "coef0: {}\n"
                    "degree: {}\n",
                    args.gamma,
                    args.coef0,
                    args.degree);
                break;
            case kernel_type::rbf:
                out << fmt::format("gamma: {}\n", args.gamma);
                break;
        }
        out << fmt::format(
            "cost: {}\n"
            "epsilon: {}\n"
            "use strings as labels: {}\n"
            "real_type: {}\n",
            args.cost,
            args.epsilon,
            args.strings_as_labels,
            detail::arithmetic_type_name<typename std::remove_reference_t<decltype(args)>::real_type>());
    }, params.base_params);
    return out << fmt::format(
               "rho: {}\n"
               "input file (data set): '{}'\n"
               "input file (model): '{}'\n"
               "output file (prediction): '{}'\n",
               params.rho,
               params.input_filename,
               params.model_filename,
               params.predict_filename);
}

}  // namespace plssvm
