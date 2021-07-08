// #include "CSVM.hpp"
#include "mocks/CSVM.hpp"
#include <gtest/gtest.h>

#include <filesystem>
#include <iostream>
#include <random>
#include <string>
#include <unistd.h>

#include "plssvm/exceptions/exceptions.hpp"

#include "plssvm/backends/OpenMP/OpenMP_CSVM.hpp"
#include <plssvm/backends/OpenMP/svm-kernel.hpp>
#include <plssvm/kernel_types.hpp>

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    #include "manager/configuration.hpp"
    #include "manager/device.hpp"
    #include "manager/manager.hpp"
    #include <plssvm/backends/OpenCL/DevicePtrOpenCL.hpp>
    #include <stdexcept>

    #include "manager/apply_arguments.hpp"
    #include "manager/run_kernel.hpp"
#endif

TEST(IO, libsvmFormat) {
    MockCSVM csvm(1., 1., plssvm::kernel_type::linear, 1., 1., 1., false);
    csvm.parse_libsvm(TESTPATH "/data/5x4.libsvm");  //TODO: add comments etc to libsvm test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    std::vector<std::vector<real_t>> expected{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { -0.20981208921241892, 0.60276937379453293, -0.13086851759108944, 0.10805254527169827 },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(IO, sparselibsvmFormat) {
    MockCSVM csvm(1., 1., plssvm::kernel_type::linear, 1., 1., 1., false);
    csvm.parse_libsvm(TESTPATH "/data/5x4.sparse.libsvm");  //TODO: add comments etc to libsvm test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    std::vector<std::vector<real_t>> expected{
        { 0., 0., 0., 0. },
        { 0., 0., 0.51687296029754564, 0. },
        { 0., 1.01405596624706053, 0., 0. },
        { 0., 0.60276937379453293, 0., -0.13086851759108944 },
        { 0., 0., 0.298499933047586044, 0. },
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(IO, arffFormat) {
    MockCSVM csvm(1., 1., plssvm::kernel_type::linear, 1., 1., 1., false);
    csvm.parse_arff(TESTPATH "/data/5x4.arff");  //TODO: add comments etc to arff test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_EQ(csvm.get_data().size(), csvm.get_num_data_points());
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        EXPECT_EQ(csvm.get_data()[i].size(), csvm.get_num_features()) << "datapoint: " << i;
    }

    std::vector<std::vector<real_t>> expected{
        { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
        { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
        { 0.57650218263054642, 1.01405596624706053, 0.13009428079760464, 0.7261913886869387 },
        { 0., 0.60276937379453293, -0.13086851759108944, 0. },
        { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 },
    };
    for (int i = 0; i < csvm.get_num_data_points(); i++) {
        for (int j = 0; j < csvm.get_num_features(); j++) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[i][j], expected[i][j]) << "datapoint: " << i << " feature: " << j;
        }
    }
}

TEST(IO, arffParserGamma) {
    MockCSVM csvm(1., 1., plssvm::kernel_type::linear, 1., 1., 1., false);
    csvm.parse_arff(TESTPATH "/data/5x4.arff");  //TODO: add comments etc to arff test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_FLOAT_EQ(1.0, csvm.get_gamma());

    MockCSVM csvm_gammazero(1., 1., plssvm::kernel_type::linear, 1., 0, 1., false);
    csvm_gammazero.parse_arff(TESTPATH "/data/5x4.arff");  //TODO: add comments etc to arff test file
    EXPECT_EQ(csvm_gammazero.get_num_data_points(), 5);
    EXPECT_EQ(csvm_gammazero.get_num_features(), 4);
    EXPECT_FLOAT_EQ(1.0 / csvm_gammazero.get_num_features(), csvm_gammazero.get_gamma());
}

TEST(IO, libsvmParserGamma) {
    MockCSVM csvm(1., 1., plssvm::kernel_type::linear, 1., 1., 1., false);
    csvm.parse_libsvm(TESTPATH "/data/5x4.libsvm");  //TODO: add comments etc to arff test file
    ASSERT_EQ(csvm.get_num_data_points(), 5);
    ASSERT_EQ(csvm.get_num_features(), 4);
    ASSERT_FLOAT_EQ(1.0, csvm.get_gamma());

    MockCSVM csvm_gammazero(1., 1., plssvm::kernel_type::linear, 1., 0, 1., false);
    csvm_gammazero.parse_libsvm(TESTPATH "/data/5x4.libsvm");  //TODO: add comments etc to arff test file
    EXPECT_EQ(csvm_gammazero.get_num_data_points(), 5);
    EXPECT_EQ(csvm_gammazero.get_num_features(), 4);
    EXPECT_FLOAT_EQ(1.0 / csvm_gammazero.get_num_features(), csvm_gammazero.get_gamma());
}

TEST(IO, writeModel) {
    MockCSVM csvm(1., 0.001, plssvm::kernel_type::linear, 3.0, 0.0, 0.0, false);
    csvm.parse_libsvm(TESTPATH "/data/5x4.libsvm");  //TODO: add comments etc to arff test file
    std::string model1 = std::tmpnam(nullptr);
    csvm.write_model(model1);
    std::ifstream model1_ifs(model1);
    std::string genfile1((std::istreambuf_iterator<char>(model1_ifs)),
                         std::istreambuf_iterator<char>());
    remove(model1.c_str());

    EXPECT_THAT(genfile1, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv 0+\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV"));

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    std::string model2 = std::tmpnam(nullptr);  // TODO: only if openmp backend is available
    MockOpenMP_CSVM csvm2(1., 0.001, plssvm::kernel_type::linear, 3.0, 0.0, 0.0, false);
    std::string testfile = TESTPATH "/data/5x4.libsvm";
    csvm2.learn(testfile, model2);

    std::ifstream model2_ifs(model2);
    std::string genfile2((std::istreambuf_iterator<char>(model2_ifs)),
                         std::istreambuf_iterator<char>());
    remove(model2.c_str());

    EXPECT_THAT(genfile2, testing::ContainsRegex("^svm_type c_svc\nkernel_type [(linear),(polynomial),(rbf)]+\nnr_class 2\ntotal_sv [1-9][0-9]*\nrho [-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?\nlabel 1 -1\nnr_sv [0-9]+ [0-9]+\nSV\n( *[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+)?( +[0-9]+:[-+]?[0-9]*.?[0-9]+([eE][-+]?[0-9]+))+ *\n*)+"));
#else
    #pragma message("Ignore OpenMP backend test")
#endif
}

TEST(IO, libsvmFormatIllFormed) {
    MockCSVM csvm(1., 1., plssvm::kernel_type::linear, 1., 1., 1., false);
    EXPECT_THROW(csvm.parse_libsvm(TESTPATH "/data/5x4.arff");, plssvm::invalid_file_format_exception);
}

TEST(IO, arffFormatIllFormed) {
    MockCSVM csvm(1., 1., plssvm::kernel_type::linear, 1., 1., 1., false);
    EXPECT_THROW(csvm.parse_arff(TESTPATH "/data/5x4.libsvm");, plssvm::invalid_file_format_exception);
}

TEST(IO, libsvmNoneExistingFile) {
    MockCSVM csvm(1., 1., plssvm::kernel_type::linear, 1., 1., 1., false);
    EXPECT_THROW(csvm.parse_libsvm(TESTPATH "/data/5x5.ar");, plssvm::file_not_found_exception);
}

TEST(IO, arffNoneExistingFile) {
    MockCSVM csvm(1., 1., plssvm::kernel_type::linear, 1., 1., 1., false);
    EXPECT_THROW(csvm.parse_arff(TESTPATH "/data/5x5.lib");, plssvm::file_not_found_exception);
}

TEST(kernel, linear) {
    const real_t degree = 0.0;
    const real_t gamma = 0.0;
    const real_t coef0 = 0.0;
    const size_t size = 512;
    std::vector<real_t> x1(size);
    std::vector<real_t> x2(size);
    real_t correct = 0;
    std::generate(x1.begin(), x1.end(), std::rand);
    std::generate(x2.begin(), x2.end(), std::rand);
    for (size_t i = 0; i < size; ++i) {
        correct += x1[i] * x2[i];
    }

    MockCSVM csvm(1., 0.001, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    real_t result = csvm.kernel_function(x1, x2);
    real_t result2 = csvm.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result);
    EXPECT_DOUBLE_EQ(correct, result2);

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    MockOpenMP_CSVM csvm_OpenMP(1., 0.001, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    real_t result_OpenMP = csvm_OpenMP.kernel_function(x1, x2);
    real_t result2_OpenMP = csvm_OpenMP.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_OpenMP);
    EXPECT_DOUBLE_EQ(correct, result2_OpenMP);
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    MockOpenCL_CSVM csvm_OpenCL(1., 0.001, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    real_t result_OpenCL = csvm_OpenCL.kernel_function(x1, x2);
    real_t result2_OpenCL = csvm_OpenCL.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_OpenCL);
    EXPECT_DOUBLE_EQ(correct, result2_OpenCL);
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    MockCUDA_CSVM csvm_CUDA(1., 0.001, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    real_t result_CUDA = csvm_CUDA.kernel_function(x1, x2);
    real_t result2_CUDA = csvm_CUDA.kernel_function(x1.data(), x2.data(), size);

    EXPECT_DOUBLE_EQ(correct, result_CUDA);
    EXPECT_DOUBLE_EQ(correct, result2_CUDA);
#endif
}

TEST(CSVM, transform_data) {
    MockCSVM csvm(1., 0.001, plssvm::kernel_type::linear, 3.0, 0.0, 0.0, false);
    csvm.parse_libsvm(TESTPATH "/data/5x4.libsvm");
    std::vector<real_t> result0 = csvm.transform_data(0);
    std::vector<real_t> result10 = csvm.transform_data(10);

    EXPECT_EQ(result0.size(), (csvm.get_num_data_points() - 1) * csvm.get_num_features());  //TODO: nochmal ünerlegen ob -1 wirklich passt (sollte eigentlich)
    EXPECT_EQ(result10.size(), (csvm.get_num_data_points() - 1 + 10) * csvm.get_num_features());

    for (size_t datapoint = 0; datapoint < csvm.get_num_data_points() - 1; ++datapoint) {
        for (size_t feature = 0; feature < csvm.get_num_features(); ++feature) {
            EXPECT_DOUBLE_EQ(csvm.get_data()[datapoint][feature], result0[datapoint + feature * (csvm.get_num_data_points() - 1)]) << "datapoint: " << datapoint << " feature: " << feature << " at index: " << datapoint + feature * (csvm.get_num_data_points() - 1);
            EXPECT_DOUBLE_EQ(csvm.get_data()[datapoint][feature], result10[datapoint + feature * (csvm.get_num_data_points() - 1 + 10)]) << "datapoint: " << datapoint << " feature: " << feature << " at index: " << datapoint + feature * (csvm.get_num_data_points() - 1 + 10);
        }
    }
}

TEST(learn, comapre_backends) {
    const real_t degree = 0.0;
    const real_t gamma = 0.0;
    const real_t coef0 = 0.0;
    const real_t eps = 0.001;
    std::vector<std::vector<real_t>> alphas;
    std::vector<std::string> svms;

    std::vector<real_t> biass;
    std::vector<real_t> QA_costs;

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    MockOpenMP_CSVM csvm_OpenMP(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    csvm_OpenMP.parse_libsvm(TESTPATH "/data/5x4.libsvm");
    csvm_OpenMP.loadDataDevice();
    csvm_OpenMP.learn();
    ASSERT_EQ(csvm_OpenMP.get_num_data_points(), csvm_OpenMP.alpha_.size());
    alphas.push_back(csvm_OpenMP.alpha_);
    QA_costs.push_back(csvm_OpenMP.QA_cost_);
    biass.push_back(csvm_OpenMP.bias_);
    svms.emplace_back("openmp");
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    MockOpenCL_CSVM csvm_OpenCL(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    csvm_OpenCL.parse_libsvm(TESTPATH "/data/5x4.libsvm");
    csvm_OpenCL.loadDataDevice();
    csvm_OpenCL.learn();
    ASSERT_EQ(csvm_OpenCL.get_num_data_points(), csvm_OpenCL.alpha_.size());
    alphas.push_back(csvm_OpenCL.alpha_);
    QA_costs.push_back(csvm_OpenCL.QA_cost_);
    biass.push_back(csvm_OpenCL.bias_);
    svms.emplace_back("opencl");
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    MockCUDA_CSVM csvm_CUDA(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    csvm_CUDA.parse_libsvm(TESTPATH "/data/5x4.libsvm");
    csvm_CUDA.loadDataDevice();
    csvm_CUDA.learn();
    ASSERT_EQ(csvm_CUDA.get_num_data_points(), csvm_CUDA.alpha_.size());
    alphas.push_back(csvm_CUDA.alpha_);
    QA_costs.push_back(csvm_CUDA.QA_cost_);
    biass.push_back(csvm_CUDA.bias_);
    svms.emplace_back("cuda");
#endif

    ASSERT_EQ(alphas.size(), biass.size());
    ASSERT_EQ(alphas.size(), QA_costs.size());
    for (size_t svm = 1; svm < alphas.size(); ++svm) {
        EXPECT_NEAR(biass[0], biass[svm], 1e-2) << "svm: " << svm;
        EXPECT_DOUBLE_EQ(QA_costs[0], QA_costs[svm]) << "svm: " << svm;
        ASSERT_EQ(alphas[0].size(), alphas[svm].size()) << "svm: " << svm;
        for (size_t index = 0; index < alphas[0].size(); ++index) {
            EXPECT_NEAR(alphas[0][index], alphas[svm][index], 1e-2) << "svm: " << svms[svm] << " index: " << index;
        }
    }
}

TEST(learn, q) {
    const real_t degree = 0.0;
    const real_t gamma = 0.0;
    const real_t coef0 = 0.0;
    const real_t eps = 0.001;
    std::vector<std::vector<real_t>> qs;
    std::vector<std::string> svms;

    MockCSVM csvm(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    csvm.parse_libsvm(TESTPATH "/data/500x200.libsvm");

    qs.emplace_back(std::vector<real_t>());

    qs[0].reserve(csvm.data_.size());
    for (int i = 0; i < csvm.data_.size() - 1; ++i) {
        qs[0].emplace_back(csvm.kernel_function(csvm.data_.back(), csvm.data_[i]));
    }
    svms.emplace_back("correct");

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    MockOpenMP_CSVM csvm_OpenMP(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    csvm_OpenMP.parse_libsvm(TESTPATH "/data/500x200.libsvm");
    csvm_OpenMP.loadDataDevice();
    qs.emplace_back(csvm_OpenMP.generate_q());
    svms.emplace_back("openmp");
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    MockOpenCL_CSVM csvm_OpenCL(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    csvm_OpenCL.parse_libsvm(TESTPATH "/data/500x200.libsvm");
    csvm_OpenCL.loadDataDevice();
    qs.emplace_back(csvm_OpenCL.generate_q());
    svms.emplace_back("opencl");
#endif
#if defined(PLSSVM_HAS_CUDA_BACKEND)
    MockCUDA_CSVM csvm_CUDA(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    csvm_CUDA.parse_libsvm(TESTPATH "/data/500x200.libsvm");
    csvm_CUDA.loadDataDevice();
    qs.emplace_back(csvm_CUDA.generate_q());
    svms.emplace_back("cuda");

#endif

    for (size_t svm = 1; svm < qs.size(); ++svm) {
        ASSERT_EQ(qs[0].size(), qs[svm].size()) << "svms[0] " << svms[0] << ", svm: " << svms[svm];
        for (size_t index = 0; index < qs[0].size(); ++index) {
            EXPECT_DOUBLE_EQ(qs[0][index], qs[svm][index]) << "svm: " << svms[svm] << " index: " << index;
        }
    }
}

TEST(learn, kernel_linear) {
    const real_t degree = 0.0;
    const real_t gamma = 0.0;
    const real_t coef0 = 0.0;
    const real_t eps = 0.001;
    std::vector<real_t> q;
    std::vector<std::vector<real_t>> rs;
    std::vector<std::string> svms;

    MockCSVM csvm(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    csvm.parse_libsvm(TESTPATH "/data/500x200.libsvm");

    q.reserve(csvm.data_.size());
    for (int i = 0; i < csvm.data_.size() - 1; ++i) {
        q.emplace_back(csvm.kernel_function(csvm.data_.back(), csvm.data_[i]));
    }
    real_t QA_cost = csvm.kernel_function(csvm.data_.back(), csvm.data_.back()) + 1 / csvm.cost_;

    real_t sgn = -1;  // TODO: +1
    const size_t dept = csvm.get_num_data_points() - 1;
    std::vector<real_t> x(dept);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<real_t> dist(-1, 2.0);

    std::generate(x.begin(), x.end(), [&]() { return dist(gen); });
    rs.emplace_back(std::vector<real_t>(dept, 0.0));

    for (int i = 0; i < dept; ++i) {
        for (int j = 0; j < dept; ++j) {
            if (i >= j) {
                real_t temp = csvm.kernel_function(&csvm.data_[i][0], &csvm.data_[j][0], csvm.get_num_features()) - q[i] - q[j] + QA_cost;
                if (i == j) {
                    rs[0][i] += (temp + 1 / csvm.cost_) * x[i] * sgn;
                } else {
                    rs[0][i] += (temp) *x[j] * sgn;
                    rs[0][j] += (temp) *x[i] * sgn;
                }
            }
        }
    }

    svms.emplace_back("correct");

#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    rs.emplace_back(std::vector<real_t>(dept, 0.0));
    plssvm::kernel_linear(rs.back(), csvm.data_, &csvm.data_.back()[0], q.data(), rs.back(), x.data(), csvm.get_num_features(), QA_cost, 1 / csvm.cost_, sgn);
    svms.emplace_back("openmp");
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    const size_t boundary_size = plssvm::THREADBLOCK_SIZE * plssvm::INTERNALBLOCK_SIZE;
    MockOpenCL_CSVM csvm_OpenCL(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    csvm_OpenCL.parse_libsvm(TESTPATH "/data/500x200.libsvm");
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

    opencl::DevicePtrOpenCL<real_t> q_cl(devices[0], q.size());
    opencl::DevicePtrOpenCL<real_t> x_cl(devices[0], x.size());
    opencl::DevicePtrOpenCL<real_t> r_cl(devices[0], dept);
    q_cl.to_device(q);
    x_cl.to_device(x);
    r_cl.to_device(std::vector<real_t>(dept, 0.0));
    q_cl.resize(dept + boundary_size);
    x_cl.resize(dept + boundary_size);
    r_cl.resize(dept + boundary_size);
    const int Ncols = csvm_OpenCL.get_num_features();
    const int Nrows = dept + plssvm::THREADBLOCK_SIZE * plssvm::INTERNALBLOCK_SIZE;

    opencl::apply_arguments(kernel, q_cl.get(), r_cl.get(), x_cl.get(), csvm_OpenCL.data_cl[0].get(), QA_cost, 1 / csvm_OpenCL.cost_, Ncols, Nrows, static_cast<int>(sgn), 0, Ncols);
    std::vector<size_t> grid_size{ static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(plssvm::THREADBLOCK_SIZE * plssvm::INTERNALBLOCK_SIZE))),
                                   static_cast<size_t>(ceil(static_cast<real_t>(dept) / static_cast<real_t>(plssvm::THREADBLOCK_SIZE * plssvm::INTERNALBLOCK_SIZE))) };
    grid_size[0] *= plssvm::THREADBLOCK_SIZE;
    grid_size[1] *= plssvm::THREADBLOCK_SIZE;
    std::vector<size_t> block_size{ plssvm::THREADBLOCK_SIZE, plssvm::THREADBLOCK_SIZE };
    opencl::run_kernel_2d_timed(devices[0], kernel, grid_size, block_size);

    r_cl.resize(dept);
    std::vector<real_t> r(dept);
    r_cl.from_device(r);
    rs.emplace_back(r);
    svms.emplace_back("opencl");
#endif

    // #endif
    // #if defined(PLSSVM_HAS_CUDA_BACKEND) //TODO: Test for CUDA kernel
    //     MockCUDA_CSVM csvm_CUDA(1., eps, plssvm::kernel_type::linear, degree, gamma, coef0, false);
    //     csvm_CUDA.libsvmParser(TESTPATH "/data/5x4.libsvm");
    //     csvm_CUDA.loadDataDevice();
    //     svms.emplace_back("cuda");

    // #endif

    for (size_t svm = 1; svm < rs.size(); ++svm) {
        ASSERT_EQ(rs[0].size(), rs[svm].size()) << "svms[0] " << svms[0] << ", svm: " << svms[svm];
        for (size_t index = 0; index < rs[0].size(); ++index) {
            EXPECT_NEAR(rs[0][index], rs[svm][index], 1e-8) << "svm: " << svms[svm] << " index: " << index;
        }
    }
}