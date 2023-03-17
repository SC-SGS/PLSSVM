#include "plssvm/csvm.hpp"

#include "utility.hpp"  // check_kwargs_for_correctness, convert_kwargs_to_parameter

#include "pybind11/pybind11.h"  // py::module_, py::class_, py::kwargs, py::overload_cast, py::const_
#include "pybind11/stl.h"       // support for STL types

#include <string>  // std::string

namespace py = pybind11;

void init_csvm(py::module_ &m) {
    using real_type = double;
    using label_type = std::string;

    // TODO: rewrite py::kwargs with std::optional?

    py::class_<plssvm::csvm>(m, "csvm")
        .def("get_params", &plssvm::csvm::get_params)
        .def("set_params", [](plssvm::csvm &self, const plssvm::parameter &params) {
            self.set_params(params);
        })
        .def("set_params", [](plssvm::csvm &self, py::kwargs args) {
            // check named arguments
            check_kwargs_for_correctness(args, { "kernel_type", "degree", "gamma", "coef0", "cost" });
            // convert kwargs to parameter and update csvm internal parameter
            self.set_params(convert_kwargs_to_parameter(args, self.get_params()));
        })
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
        })
        .def("predict", &plssvm::csvm::predict<real_type, label_type>)
        .def("score", py::overload_cast<const plssvm::model<real_type, label_type> &>(&plssvm::csvm::score<real_type, label_type>, py::const_))
        .def("score", py::overload_cast<const plssvm::model<real_type, label_type> &, const plssvm::data_set<real_type, label_type> &>(&plssvm::csvm::score<real_type, label_type>, py::const_));
}