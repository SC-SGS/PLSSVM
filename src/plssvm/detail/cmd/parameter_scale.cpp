/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/cmd/parameter_scale.hpp"

#include "plssvm/constants.hpp"  // plssvm::verbose

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
           ("s,save_filename", "the file to which the scaling factors should be saved", cxxopts::value<decltype(save_filename)>())
           ("r,restore_filename", "the file to which the scaling factors should be saved", cxxopts::value<decltype(save_filename)>())
           ("use_strings_as_labels", "use strings as labels instead of plane numbers", cxxopts::value<decltype(strings_as_labels)>()->default_value(fmt::format("{}", strings_as_labels)))
           ("use_float_as_real_type", "use floats as real types instead of doubles", cxxopts::value<decltype(float_as_real_type)>()->default_value(fmt::format("{}", float_as_real_type)))
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

   // parse the lowest allowed value
   lower = result["lower"].as<decltype(lower)>();

   // parse the highest allowed value
   upper = result["upper"].as<decltype(upper)>();

   // parse the file format
   format = result["format"].as<decltype(format)>();

   // parse whether strings should be used as labels
   strings_as_labels = result["use_strings_as_labels"].as<decltype(strings_as_labels)>();

   // parse whether floats should be used as real_type
   float_as_real_type = result["use_float_as_real_type"].as<decltype(float_as_real_type)>();

   // parse whether output is quiet or not
   plssvm::verbose = !plssvm::verbose;

   // parse input data filename
   if (!result.count("input")) {
       fmt::print(stderr, "Error missing input file!\n");
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

   if (result.count("save_filename") && result.count("restore_filename")) {
       fmt::print(stderr, "Error cannot use -s (--save_filename) and -r (--restore_filename) simultaneously!");
       fmt::print("{}", options.help());
       std::exit(EXIT_FAILURE);
   }

   if (result.count("save_filename")) {
       save_filename = result["save_filename"].as<decltype(save_filename)>();
   }

   if (result.count("restore_filename")) {
       if (result.count("lower") || result.count("upper")) {
           std::clog << "Warning: provided -l (--lower) and/or -u (--upper) together with -r (--restore_filename); ignoring -l/-u" << std::endl;
       }
       restore_filename = result["restore_filename"].as<decltype(restore_filename)>();
   }

}

std::ostream &operator<<(std::ostream &out, const parameter_scale &params) {
   return out << fmt::format(
              "lower: {}\n"
              "upper: {}\n"
              "use strings as labels: {}\n"
              "use float as real type: {}\n"
              "output file format: {}\n"
              "input file (data set): '{}'\n"
              "output file (scaled data set): '{}'\n",
              params.lower,
              params.upper,
              params.strings_as_labels,
              params.float_as_real_type,
              params.format,
              params.input_filename,
              params.scaled_filename);
}

}  // namespace plssvm