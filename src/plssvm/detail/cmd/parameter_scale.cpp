/**
* @author Alexander Van Craen
* @author Marcel Breyer
* @copyright 2018-today The PLSSVM project - All Rights Reserved
* @license This file is part of the PLSSVM project which is released under the MIT license.
*          See the LICENSE.md file in the project root for full license information.
*/

#include "plssvm/detail/cmd/parameter_scale.hpp"

#include "plssvm/backend_types.hpp"                      // plssvm::list_available_backends
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

parameter_scale::parameter_scale(int argc, char **argv) {
   cxxopts::Options options(argv[0], "LS-SVM with multiple (GPU-)backends");
   options
       .positional_help("input_file [scaled_file]")
       .show_positional_help();
   options
       .set_width(150)
       .set_tab_expansion()
       // clang-format off
       .add_options()
           ("l,lower", "lower is the lowest (minimal) value allowed in each dimension", cxxopts::value<decltype(lower)>()->default_value(fmt::format("{}", lower)))
           ("u,upper", "upper is the highest (maximal) value allowed in each dimension", cxxopts::value<decltype(upper)>()->default_value(fmt::format("{}", upper)))
           ("f,format", "the file format to output the scaled data set to", cxxopts::value<decltype(format)>()->default_value(fmt::format("{}", format)))
           ("use_strings_as_labels", "use strings as labels instead of plane numbers", cxxopts::value<bool>()->default_value("false"))
           ("use_float_as_real_type", "use floats as real types instead of doubles", cxxopts::value<bool>()->default_value("false"))
           ("q,quiet", "quiet mode (no outputs)", cxxopts::value<bool>(plssvm::verbose)->default_value(fmt::format("{}", !plssvm::verbose)))
           ("h,help", "print this helper message", cxxopts::value<bool>())
           ("input", "", cxxopts::value<decltype(input_filename)>(), "input_file")
           ("scaled", "", cxxopts::value<decltype(scaled_filename)>(), "scaled_file");
   // clang-format on

   // parse command line options
   cxxopts::ParseResult result;
   try {
       options.parse_positional({ "input", "scaled" });
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

   // parse the lowest allowed value
   lower = result["lower"].as<decltype(lower)>();

   // parse the highest allowed value
   upper = result["upper"].as<decltype(upper)>();

   // parse the file format
   format = result["format"].as<decltype(format)>();

   // parse base_params values
   std::visit([&](auto&& params) {
       // parse whether strings should be used as labels
       params.strings_as_labels = result["use_strings_as_labels"].as<decltype(params.strings_as_labels)>();

   }, base_params);

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
   if (result.count("scaled")) {
       scaled_filename = result["scaled"].as<decltype(scaled_filename)>();
   } else {
       std::filesystem::path input_path{ input_filename };
       scaled_filename = input_path.replace_filename(input_path.filename().string() + ".scaled").string();
   }
}


std::ostream &operator<<(std::ostream &out, const parameter_scale &params) {
   bool strings_as_labels;
   std::string_view real_type_string;
   std::visit([&](auto&& arg) {
       strings_as_labels = arg.strings_as_labels;
       real_type_string = detail::arithmetic_type_name<typename std::remove_reference_t<decltype(arg)>::real_type>();
   }, params.base_params);

   return out << fmt::format(
              "lower: {}\n"
              "upper: {}\n"
              "use strings as labels: {}\n"
              "real_type: {}\n"
              "output file format: {}\n"
              "input file (data set): '{}'\n"
              "output file (scaled data set): '{}'\n",
              params.lower,
              params.upper,
              strings_as_labels,
              real_type_string,
              params.format,
              params.input_filename,
              params.scaled_filename);
}

}  // namespace plssvm