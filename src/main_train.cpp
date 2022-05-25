/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Main function compiled to the `plssvm-train` executable used for training a C-SVM model.
 */

#include "plssvm/core.hpp"

#include "fmt/color.h"    // fmt::fg, fmt::color::orange
#include "fmt/core.h"     // std::format
#include "fmt/ostream.h"  // use operator<< to output enum class

#include <cstdlib>    // EXIT_SUCCESS, EXIT_FAILURE
#include <exception>  // std::exception
#include <iostream>   // std::cerr, std::clog, std::endl
#include <variant>    // std::visit


int main(int argc, char *argv[]) {
    try {
        // parse SVM parameter from command line
        plssvm::detail::cmd::parameter_train params{ argc, argv };

        // warn if kernel invocation type nd_range or hierarchical are explicitly set but SYCL isn't the current backend
        if (params.backend != plssvm::backend_type::sycl && params.sycl_kernel_invocation_type != plssvm::sycl::kernel_invocation_type::automatic) {
            std::clog << fmt::format(fmt::fg(fmt::color::orange),
                                     "WARNING: explicitly set a SYCL kernel invocation type but the current backend isn't SYCL; ignoring --sycl_kernel_invocation_type={}",
                                     params.sycl_kernel_invocation_type)
                      << std::endl;
        }
        // warn if a SYCL implementation type is explicitly set but SYCL isn't the current backend
        if (params.backend != plssvm::backend_type::sycl && params.sycl_implementation_type != plssvm::sycl::implementation_type::automatic) {
            std::clog << fmt::format(fmt::fg(fmt::color::orange),
                                     "WARNING: explicitly set a SYCL implementation type but the current backend isn't SYCL; ignoring --sycl_implementation_type={}",
                                     params.sycl_implementation_type)
                      << std::endl;
        }

        // output used parameter
        if (plssvm::verbose) {
            fmt::print("\ntask: training\n{}\n", params);
        }

        // create data set
        std::visit([&](auto&& data){
            // TODO: put code here
            using real_type = typename std::remove_reference_t<decltype(data)>::real_type;
            using label_type = typename std::remove_reference_t<decltype(data)>::label_type;

            // convert base params to correct type
            auto csvm_params = static_cast<plssvm::parameter<real_type>>(params.csvm_params);
            // create SVM
            auto svm = plssvm::make_csvm<real_type>(params.backend, params.target, csvm_params);
            // learn model
            plssvm::model<real_type, label_type> model = svm->fit(data, plssvm::epsilon = params.epsilon, plssvm::max_iter = params.max_iter);
            // save model to file
            model.save(params.model_filename);

        }, plssvm::detail::data_set_factory<false>(params));

    } catch (const plssvm::exception &e) {
        std::cerr << e.what_with_loc() << std::endl;
        return EXIT_FAILURE;
    } catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }
    return EXIT_SUCCESS;
}
