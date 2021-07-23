#include "mock_opencl_csvm.hpp"

#include "../../mock_csvm.hpp"  // mock_csvm
#include "../../utility.hpp"    // util::create_temp_file, util::gtest_expect_floating_point_eq, util::google_test::parameter_definition, util::google_test::parameter_definition_to_name

#include "../compare.hpp"                    // compare::generate_q, compare::kernel_function, compare::device_kernel_function
#include "plssvm/backends/OpenCL/csvm.hpp"   // plssvm::opencl::csvm
#include "plssvm/constants.hpp"              // plssvm::THREAD_BLOCK_SIZE
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::replace_all
#include "plssvm/kernel_types.hpp"           // plssvm::kernel_type
#include "plssvm/parameter.hpp"              // plssvm::parameter

#include "../../../src/plssvm/backends/OpenCL/manager/apply_arguments.hpp"
#include "../../../src/plssvm/backends/OpenCL/manager/configuration.hpp"
#include "../../../src/plssvm/backends/OpenCL/manager/device.hpp"
#include "../../../src/plssvm/backends/OpenCL/manager/manager.hpp"
#include "../../../src/plssvm/backends/OpenCL/manager/run_kernel.hpp"

#include "plssvm/backends/OpenCL/DevicePtrOpenCL.hpp"

#include <cmath>        // std::abs
#include <cstddef>      // std::size_t
#include <filesystem>   // std::filesystem::remove
#include <fstream>      // std::ifstream
#include <iterator>     // std::istreambuf_iterator
#include <random>       // std::random_device, std::mt19937, std::uniform_real_distribution
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

template <typename T>
class OpenCL_base : public ::testing::Test {};

using write_model_parameter_types = ::testing::Types<float, double>;
TYPED_TEST_SUITE(OpenCL_base, write_model_parameter_types);

TYPED_TEST(OpenCL_base, write_model) {
    // setup OpenCL C-SVM
    plssvm::parameter<TypeParam> params{ TEST_PATH "/data/5x4.libsvm" };
    params.print_info = false;

    mock_opencl_csvm csvm{ params };

    // create temporary model file
    std::string model_file = util::create_temp_file();

    // learn
    csvm.learn(params.input_filename, model_file);

    // read content of model file and delete it
    std::ifstream model_ifs(model_file);
    std::string file_content((std::istreambuf_iterator<char>(model_ifs)), std::istreambuf_iterator<char>());
    std::filesystem::remove(model_file);

    // check model file content for correctness
    EXPECT_THAT(file_content, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv [1-9][0-9]*\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
}

// enumerate all type and kernel combinations to test
using parameter_types = ::testing::Types<
    util::google_test::parameter_definition<float, plssvm::kernel_type::linear>,
    util::google_test::parameter_definition<double, plssvm::kernel_type::linear>>;

// generate tests for the generation of the q vector
template <typename T>
class OpenCL_generate_q : public ::testing::Test {};
TYPED_TEST_SUITE(OpenCL_generate_q, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(OpenCL_generate_q, generate_q) {
    // setup C-SVM
    plssvm::parameter<typename TypeParam::real_type> params{ TEST_FILE };
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    mock_csvm csvm{ params };
    using real_type_csvm = typename decltype(csvm)::real_type;

    // parse libsvm file and calculate q vector
    csvm.parse_libsvm(params.input_filename);
    const std::vector<real_type_csvm> correct = compare::generate_q<TypeParam::kernel>(csvm.get_data(), csvm);

    // setup OpenCL C-SVM
    mock_opencl_csvm csvm_opencl{ params };
    using real_type_csvm_opencl = typename decltype(csvm_opencl)::real_type;

    // check real_types
    ::testing::StaticAssertTypeEq<real_type_csvm, real_type_csvm_opencl>();

    // parse libsvm file and calculate q vector
    csvm_opencl.parse_libsvm(params.input_filename);
    csvm_opencl.setup_data_on_device();
    const std::vector<real_type_csvm_opencl> calculated = csvm_opencl.generate_q();

    ASSERT_EQ(correct.size(), calculated.size());
    for (std::size_t index = 0; index < correct.size(); ++index) {
        util::gtest_assert_floating_point_near(correct[index], calculated[index], fmt::format("\tindex: {}", index));
    }
}

// generate tests for the device kernel functions
template <typename T>
class OpenCL_device_kernel : public ::testing::Test {};
TYPED_TEST_SUITE(OpenCL_device_kernel, parameter_types, util::google_test::parameter_definition_to_name);

TYPED_TEST(OpenCL_device_kernel, device_kernel) {
    // setup C-SVM
    plssvm::parameter<typename TypeParam::real_type> params{ TEST_FILE };
    params.print_info = false;
    params.kernel = TypeParam::kernel;

    mock_csvm csvm{ params };
    using real_type = typename decltype(csvm)::real_type;
    using size_type = typename decltype(csvm)::size_type;

    // parse libsvm file
    csvm.parse_libsvm(params.input_filename);

    const size_type dept = csvm.get_num_data_points() - 1;
    const size_type num_features = csvm.get_num_features();

    // create x vector and fill it with random values
    std::vector<real_type> x(dept);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_type> dist(-1, 2.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

    // create correct q vector, cost and QA_cost
    const std::vector<real_type> q_vec = compare::generate_q<TypeParam::kernel>(csvm.get_data(), csvm);
    const real_type cost = csvm.get_cost();
    const real_type QA_cost = compare::kernel_function<TypeParam::kernel>(csvm.get_data().back(), csvm.get_data().back(), csvm) + 1 / cost;

    // setup OpenCL C-SVM
    mock_opencl_csvm csvm_opencl{ params };

    // parse libsvm file
    csvm_opencl.parse_libsvm(params.input_filename);

    // setup data on device
    csvm_opencl.setup_data_on_device();

    // assemble kernel name
    std::string kernel_name;
    if constexpr (TypeParam::kernel == plssvm::kernel_type::linear) {
        kernel_name = "kernel_linear";
    } else if constexpr (TypeParam::kernel == plssvm::kernel_type::polynomial) {
        kernel_name = "kernel_poly";
    } else if constexpr (TypeParam::kernel == plssvm::kernel_type::rbf) {
        kernel_name = "kernel_radial";
    }

    std::vector<opencl::device_t> &devices = csvm_opencl.manager.get_devices();
    std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/svm-kernel.cl" };
    std::string kernel_src = csvm_opencl.manager.read_src_file(kernel_src_file_name);
    if constexpr (std::is_same_v<real_type, float>) {
        csvm_opencl.manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
        plssvm::detail::replace_all(kernel_src, "real_type", "float");
    } else if constexpr (std::is_same_v<real_type, double>) {
        csvm_opencl.manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
        plssvm::detail::replace_all(kernel_src, "real_type", "double");
    }
    json::node &deviceNode =
        csvm_opencl.manager.get_configuration()["PLATFORMS"][devices[0].platformName]
                                               ["DEVICES"][devices[0].deviceName];
    json::node &kernelConfig = deviceNode["KERNELS"][kernel_name];

    kernelConfig.replaceTextAttr("INTERNAL_BLOCK_SIZE", std::to_string(plssvm::INTERNAL_BLOCK_SIZE));
    kernelConfig.replaceTextAttr("THREAD_BLOCK_SIZE", std::to_string(plssvm::THREAD_BLOCK_SIZE));
    cl_kernel kernel = csvm_opencl.manager.build_kernel(kernel_src, devices[0], kernelConfig, kernel_name);

    const size_type boundary_size = plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;
    opencl::DevicePtrOpenCL<real_type> q_cl(devices[0], q_vec.size());
    opencl::DevicePtrOpenCL<real_type> x_cl(devices[0], x.size());
    opencl::DevicePtrOpenCL<real_type> r_cl(devices[0], dept);
    q_cl.to_device(q_vec);
    x_cl.to_device(x);
    r_cl.to_device(std::vector<real_type>(dept, 0.0));
    q_cl.resize(dept + boundary_size);
    x_cl.resize(dept + boundary_size);
    r_cl.resize(dept + boundary_size);
    const int Ncols = num_features;
    const int Nrows = dept + plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE;

    std::vector<size_t> grid_size{ static_cast<size_t>(ceil(static_cast<real_type>(dept) / static_cast<real_type>(plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE))),
                                   static_cast<size_t>(ceil(static_cast<real_type>(dept) / static_cast<real_type>(plssvm::THREAD_BLOCK_SIZE * plssvm::INTERNAL_BLOCK_SIZE))) };
    std::vector<size_t> block_size{ plssvm::THREAD_BLOCK_SIZE, plssvm::THREAD_BLOCK_SIZE };
    grid_size[0] *= plssvm::THREAD_BLOCK_SIZE;
    grid_size[1] *= plssvm::THREAD_BLOCK_SIZE;

    for (const int add : { -1, 1 }) {
        std::vector<real_type> correct = compare::device_kernel_function<TypeParam::kernel>(csvm.get_data(), x, q_vec, QA_cost, cost, add, csvm);

        std::vector<real_type> result(dept, 0.0);
        opencl::apply_arguments(kernel, q_cl.get(), r_cl.get(), x_cl.get(), csvm_opencl.get_device_data()[0].get(), QA_cost, cost, Ncols, Nrows, add, 0, Ncols);
        opencl::run_kernel_2d_timed(devices[0], kernel, grid_size, block_size);

        r_cl.resize(dept);
        r_cl.from_device(result);

        r_cl.resize(dept + boundary_size);
        r_cl.to_device(std::vector<real_type>(dept + boundary_size, 0.0));

        ASSERT_EQ(correct.size(), result.size()) << "add: " << add;
        for (size_t index = 0; index < correct.size(); ++index) {
            util::gtest_assert_floating_point_near(correct[index], result[index], fmt::format("\tindex: {}, add: {}", index, add));
        }
    }
}
