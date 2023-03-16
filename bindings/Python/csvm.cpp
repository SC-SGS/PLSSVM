#include "plssvm/csvm.hpp"

#include "utility.hpp"  // check_kwargs_for_correctness

#include "pybind11/pybind11.h"  // py::module, py::class_, py::kwargs, py::overload_cast, py::const_
#include "pybind11/stl.h"       // support for STL types

#include <string>  // std::string

namespace py = pybind11;

void init_csvm(py::module &m) {
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

            plssvm::parameter params = self.get_params();
            if (args.contains("kernel_type")) {
                params.kernel_type = args["kernel_type"].cast<plssvm::kernel_function_type>();
            }
            if (args.contains("degree")) {
                params.degree = args["degree"].cast<int>();
            }
            if (args.contains("gamma")) {
                params.gamma = args["gamma"].cast<real_type>();
            }
            if (args.contains("coef0")) {
                params.coef0 = args["coef0"].cast<real_type>();
            }
            if (args.contains("cost")) {
                params.cost = args["cost"].cast<real_type>();
            }
            self.set_params(params);
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