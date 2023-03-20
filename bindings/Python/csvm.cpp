#include "plssvm/csvm.hpp"
#include "plssvm/csvm_factory.hpp"

#include "utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::kwargs, py::overload_cast, py::const_
#include "pybind11/stl.h"       // support for STL types

#include <string>  // std::string

namespace py = pybind11;

void init_csvm(py::module_ &m) {
    using real_type = double;
    using label_type = std::string;

    py::module_ pure_virtual_model = m.def_submodule("__pure_virtual");

    py::class_<plssvm::csvm> pycsvm(pure_virtual_model, "__pure_virtual_base_csvm");
    pycsvm.def("get_params", &plssvm::csvm::get_params, "get the parameter used for this SVM")
        .def("set_params", [](plssvm::csvm &self, const plssvm::parameter &params) {
            self.set_params(params);
        }, "update the parameter used for this SVM using a plssvm.Parameter object")
        .def("set_params", [](plssvm::csvm &self, py::kwargs args) {
            // check named arguments
            check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
            // convert kwargs to parameter and update csvm internal parameter
            self.set_params(convert_kwargs_to_parameter(args, self.get_params()));
        }, "update the parameter used for this SVM using keyword arguments")
        .def("fit", [](const plssvm::csvm &self, const plssvm::data_set<real_type, label_type> &data, py::kwargs args) {
            // check named arguments
            check_kwargs_for_correctness(args, { "epsilon", "max_iter" });

            if (args.contains("epsilon") && args.contains("max_iter")) {
                return self.fit(data, plssvm::epsilon = args["epsilon"].cast<real_type>(), plssvm::max_iter = args["max_iter"].cast<unsigned long long>());
            } else if (args.contains("epsilon")) {
                return self.fit(data, plssvm::epsilon = args["epsilon"].cast<real_type>());
            } else if (args.contains("max_iter")) {
                return self.fit(data, plssvm::max_iter = args["max_iter"].cast<unsigned long long>());
            } else {
                return self.fit(data);
            }
        }, "fit a model using the current SVM on the provided data")
        .def("predict", &plssvm::csvm::predict<real_type, label_type>, "predict the labels for a data set using a previously learned model")
        .def("score", py::overload_cast<const plssvm::model<real_type, label_type> &>(&plssvm::csvm::score<real_type, label_type>, py::const_), "calculate the accuracy of the model")
        .def("score", py::overload_cast<const plssvm::model<real_type, label_type> &, const plssvm::data_set<real_type, label_type> &>(&plssvm::csvm::score<real_type, label_type>, py::const_), "calculate the accuracy of a data set using the model");

    // bind plssvm::make_csvm factory function to a "generic" Python csvm class
    py::class_<plssvm::csvm>(m, "Csvm", pycsvm, py::module_local())
        .def(py::init([](py::kwargs args) {
            // check named arguments
            check_kwargs_for_correctness(args, { "backend", "target_platform", "kernel_type", "degree", "gamma", "coef0", "cost", "sycl_implementation_type", "sycl_kernel_invocation_type" });
            // convert kwargs to parameter and update csvm internal parameter
            const plssvm::parameter params = convert_kwargs_to_parameter(args);
            // backend type
            plssvm::backend_type backend = plssvm::backend_type::automatic;
            if (args.contains("backend")) {
                backend = args["backend"].cast<plssvm::backend_type>();
            }
            // target platform
            plssvm::target_platform target = plssvm::target_platform::automatic;
            if (args.contains("target_platform")) {
                target = args["target_platform"].cast<plssvm::target_platform>();
            }
            if (backend == plssvm::backend_type::sycl) {
                // sycl specific flags
                plssvm::sycl::implementation_type impl_type = plssvm::sycl::implementation_type::automatic;
                if (args.contains("sycl_implementation_type")) {
                    impl_type = args["sycl_implementation_type"].cast<plssvm::sycl::implementation_type>();
                }
                plssvm::sycl::kernel_invocation_type invocation_type = plssvm::sycl::kernel_invocation_type::automatic;
                if (args.contains("sycl_kernel_invocation_type")) {
                    invocation_type = args["sycl_kernel_invocation_type"].cast<plssvm::sycl::kernel_invocation_type>();
                }

                return plssvm::make_csvm(backend, target, params, plssvm::sycl_implementation_type = impl_type, plssvm::sycl_kernel_invocation_type = invocation_type);
            } else {
                return plssvm::make_csvm(backend, target, params);
            }
        }), "create the 'best' SVM for the available (or provided) backends and target platforms");
}