/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*/

#include "plssvm/detail/cmd/parameter_predict.hpp"

#include "plssvm/backend_types.hpp"                      // plssvm::list_available_backends
#include "plssvm/backends/SYCL/implementation_type.hpp"  // plssvm::sycl::list_available_sycl_implementations
#include "plssvm/constants.hpp"                          // plssvm::verbose
#include "plssvm/target_platforms.hpp"                   // plssvm::list_available_target_platforms

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
           ("b,backend", fmt::format("choose the backend: {}", fmt::join(list_available_backends(), "|")), cxxopts::value<backend_type>()->default_value(fmt::format("{}", backend)))
           ("p,target_platform", fmt::format("choose the target platform: {}", fmt::join(list_available_target_platforms(), "|")), cxxopts::value<target_platform>()->default_value(fmt::format("{}", target)))
#if defined(PLSSVM_HAS_SYCL_BACKEND)
           ("sycl_implementation_type", fmt::format("choose the SYCL implementation to be used in the SYCL backend: {}", fmt::join(sycl::list_available_sycl_implementations(), "|")), cxxopts::value<sycl::implementation_type>()->default_value(fmt::format("{}", sycl_implementation_type)))
#endif
           ("use_strings_as_labels", "use strings as labels instead of plane numbers", cxxopts::value<decltype(strings_as_labels)>()->default_value(fmt::format("{}", strings_as_labels)))
           ("use_float_as_real_type", "use floats as real types instead of doubles", cxxopts::value<decltype(float_as_real_type)>()->default_value(fmt::format("{}", float_as_real_type)))
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
   backend = result["backend"].as<decltype(backend)>();

   // parse target_platform and cast the value to the respective enum
   target = result["target_platform"].as<decltype(target)>();

#if defined(PLSSVM_HAS_SYCL_BACKEND)
   // parse SYCL implementation used in the SYCL backend
   sycl_implementation_type = result["sycl_implementation_type"].as<decltype(sycl_implementation_type)>();
#endif

   // parse whether strings should be used as labels
   strings_as_labels = result["use_strings_as_labels"].as<decltype(strings_as_labels)>();

   // parse whether output is quiet or not
   plssvm::verbose = !plssvm::verbose;

   // parse test data filename
   if (!result.count("test")) {
       fmt::print(stderr, "Error missing test file!\n");
       fmt::print("{}", options.help());
       std::exit(EXIT_FAILURE);
   }
   input_filename = result["test"].as<decltype(input_filename)>();

   // parse model filename
   if (!result.count("model")) {
       fmt::print(stderr, "Error missing model file!\n");
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
    return out << fmt::format(
        "use strings as labels: {}\n"
        "use float as real type instead of double: {}\n"
        "input file (data set): '{}'\n"
        "input file (model): '{}'\n"
        "output file (prediction): '{}'\n",
        params.strings_as_labels,
        params.float_as_real_type,
        params.input_filename,
        params.model_filename,
        params.predict_filename);
}

}  // namespace plssvm
