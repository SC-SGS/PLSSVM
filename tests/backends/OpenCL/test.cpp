#include "../MockCSVM.hpp"
#include "../compare.hpp"
#include "MockOpenCL_CSVM.hpp"
#include "manager/apply_arguments.hpp"
#include "manager/configuration.hpp"
#include "manager/device.hpp"
#include "manager/manager.hpp"
#include "manager/run_kernel.hpp"
#include "plssvm/backends/OpenCL/DevicePtrOpenCL.hpp"

#include <random>

TEST(IO, writeModel) {
    std::string model = std::tmpnam(nullptr);
    MockOpenCL_CSVM csvm2(1., 0.001, plssvm::kernel_type::linear, 3.0, 0.0, 0.0, false);
    std::string testfile = TESTPATH "/data/5x4.libsvm";
    csvm2.learn(testfile, model);

    std::ifstream model_ifs(model);
    std::string genfile2((std::istreambuf_iterator<char>(model_ifs)),
                         std::istreambuf_iterator<char>());
    remove(model.c_str());

    EXPECT_THAT(genfile2, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv [1-9][0-9]*\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
}

TEST(learn, q) {
    std::vector correct = generate_q(TESTFILE);

    MockOpenCL_CSVM csvm_OpenCL;
    csvm_OpenCL.libsvmParser(TESTFILE);
    csvm_OpenCL.loadDataDevice();
    std::vector test = csvm_OpenCL.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], test[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TEST(kernel, linear) {
    const size_t size = 512;
    std::vector<real_t> x1(size);
    std::vector<real_t> x2(size);
    std::generate(x1.begin(), x1.end(), std::rand);
    std::generate(x2.begin(), x2.end(), std::rand);
    real_t correct = linear_kernel(x1, x2);

    MockOpenCL_CSVM csvm_OpenCL(1., 0.001, plssvm::kernel_type::linear);
    real_t result_OpenCL = csvm_OpenCL.kernel_function(x1, x2);
    real_t result2_OpenCL = csvm_OpenCL.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_OpenCL);
    EXPECT_DOUBLE_EQ(correct, result2_OpenCL);
}

TEST(learn, q_linear) {
    MockCSVM csvm;
    csvm.libsvmParser(TESTFILE);
    std::vector<real_t> correct = q<plssvm::kernel_type::linear>(csvm.get_data());

    MockOpenCL_CSVM csvm_OpenCL(1., 0.001, plssvm::kernel_type::linear);
    csvm_OpenCL.libsvmParser(TESTFILE);
    csvm_OpenCL.loadDataDevice();
    std::vector<real_t> test = csvm_OpenCL.generate_q();

    ASSERT_EQ(correct.size(), test.size());
    for (size_t index = 0; index < correct.size(); ++index) {
        EXPECT_NEAR(correct[index], test[index], std::abs(correct[index] * 1e-10)) << " index: " << index;
    }
}

TEST(learn, kernel_linear) {
    MockCSVM csvm;
    csvm.libsvmParser(TESTFILE);

    const size_t dept = csvm.get_num_data_points() - 1;

    std::vector<real_t> x(dept);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_t> dist(-1, 2.0);
    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });

    const std::vector<real_t> q_ = q<plssvm::kernel_type::linear>(csvm.get_data());

    const real_t cost = csvm.cost;

    const real_t QA_cost = linear_kernel(csvm.data.back(), csvm.data.back()) + 1 / cost;

    const size_t boundary_size = plssvm::THREADBLOCK_SIZE * plssvm::INTERNALBLOCK_SIZE;
    MockOpenCL_CSVM csvm_OpenCL(1., 0.001, plssvm::kernel_type::linear);
    csvm_OpenCL.libsvmParser(TESTFILE);
    csvm_OpenCL.loadDataDevice();

    std::vector<opencl::device_t> &devices = csvm_OpenCL.manager.get_devices();

    std::string kernel_src_file_name{ "../src/plssvm/backends/OpenCL/kernels/svm-kernel-linear_debug.cl" };
    std::string kernel_src = csvm_OpenCL.manager.read_src_file(kernel_src_file_name);
    if (*typeid(real_t).name() == 'f') {
        csvm_OpenCL.manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "float");
    } else if (*typeid(real_t).name() == 'd') {
        csvm_OpenCL.manager.parameters.replaceTextAttr("INTERNAL_PRECISION", "double");
    }
    json::node &deviceNode =
        csvm_OpenCL.manager.get_configuration()["PLATFORMS"][devices[0].platformName]
                                               ["DEVICES"][devices[0].deviceName];
    json::node &kernelConfig = deviceNode["KERNELS"]["kernel_linear"];

    kernelConfig.replaceTextAttr("INTERNALBLOCK_SIZE", std::to_string(plssvm::INTERNALBLOCK_SIZE));
    kernelConfig.replaceTextAttr("THREADBLOCK_SIZE", std::to_string(plssvm::THREADBLOCK_SIZE));
    cl_kernel kernel = csvm_OpenCL.manager.build_kernel(kernel_src, devices[0], kernelConfig, "kernel_linear");

    opencl::DevicePtrOpenCL<real_t> q_cl(devices[0], q_.size());
    opencl::DevicePtrOpenCL<real_t> x_cl(devices[0], x.size());
    opencl::DevicePtrOpenCL<real_t> r_cl(devices[0], dept);
    q_cl.to_device(q_);
    x_cl.to_device(x);
    r_cl.to_device(std::vector<real_t>(dept, 0.0));
    q_cl.resize(dept + boundary_size);
    x_cl.resize(dept + boundary_size);
    r_cl.resize(dept + boundary_size);
    const int Ncols = csvm_OpenCL.get_num_features();
    const int Nrows = dept + plssvm::THREADBLOCK_SIZE * plssvm::INTERNALBLOCK_SIZE;

    std::vector<size_t> grid_size{ static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(plssvm::THREADBLOCK_SIZE * plssvm::INTERNALBLOCK_SIZE))),
                                   static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(plssvm::THREADBLOCK_SIZE * plssvm::INTERNALBLOCK_SIZE))) };
    std::vector<size_t> block_size{ plssvm::THREADBLOCK_SIZE, plssvm::THREADBLOCK_SIZE };
    grid_size[0] *= plssvm::THREADBLOCK_SIZE;
    grid_size[1] *= plssvm::THREADBLOCK_SIZE;

    for (const real_t sgn : { -1.0 }) {  // TODO: fix bug 1
        std::vector<real_t> correct = kernel_linear_function(csvm.get_data(), x, q_, sgn, QA_cost, cost);

        std::vector<real_t> result(dept, 0.0);
        opencl::apply_arguments(kernel, q_cl.get(), r_cl.get(), x_cl.get(), csvm_OpenCL.data_cl[0].get(), QA_cost, 1 / csvm_OpenCL.cost, Ncols, Nrows, static_cast<int>(sgn), 0, Ncols);
        opencl::run_kernel_2d_timed(devices[0], kernel, grid_size, block_size);

        r_cl.resize(dept);
        r_cl.from_device(result);

        ASSERT_EQ(correct.size(), result.size()) << "sgn: " << sgn;
        for (size_t index = 0; index < correct.size(); ++index) {
            EXPECT_NEAR(correct[index], result[index], std::abs(correct[index] * 1e-6)) << " index: " << index << " sgn: " << sgn; // TODO: nochmal anschauen Nur 6 Stellen genau ist komisch
        }
    }
}