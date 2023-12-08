/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the base C-SVM functions through its mock class.
 */

#include "plssvm/csvm.hpp"
#include "mock_csvm.hpp"  // mock_csvm

#include "plssvm/classification_types.hpp"   // plssvm::classification_type
#include "plssvm/constants.hpp"              // plssvm::real_type, plssvm::PADDING_SIZE
#include "plssvm/core.hpp"                   // necessary for type_traits, plssvm::csvm_backend_exists, plssvm::csvm_backend_exists_v
#include "plssvm/data_set.hpp"               // plssvm::data_set
#include "plssvm/detail/simple_any.hpp"      // plssvm::detail::simple_any
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::invalid_parameter_exception
#include "plssvm/kernel_function_types.hpp"  // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                 // plssvm::aos_matrix
#include "plssvm/model.hpp"                  // plssvm::model
#include "plssvm/parameter.hpp"              // plssvm::parameter
#include "plssvm/solver_types.hpp"           // plssvm::solver_type
#include "plssvm/target_platforms.hpp"       // plssvm::target_platform

#include "custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_INCLUSIVE_RANGE
#include "naming.hpp"              // naming::parameter_definition_to_name
#include "types_to_test.hpp"       // util::label_type_classification_type_gtest
#include "utility.hpp"             // util::{redirect_output, temporary_file, instantiate_template_file, get_num_classes, calculate_number_of_classifiers,
                                   // generate_random_matrix, get_correct_data_file_labels}

#include "gmock/gmock.h"           // EXPECT_CALL, ::testing::{An, Between, Return}
#include "gtest/gtest-matchers.h"  // EXPECT_THAT, ::testing::HasSubstr
#include "gtest/gtest.h"           // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_THAT,
#include <cstddef>                 // std::size_t
#include <iostream>                // std::clog
#include <string>                  // std::string
#include <tuple>                   // std::ignore
#include <vector>                  // std::vector

class BaseCSVM : public ::testing::Test {};

TEST(BaseCSVM, default_construct_from_parameter) {
    // create mock_csvm using the default parameter (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{};

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), plssvm::parameter{});
}
TEST(BaseCSVM, construct_from_parameter) {
    // create parameter class
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.01 } };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ params };

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}

TEST(BaseCSVM, construct_from_parameter_invalid_kernel_type) {
    // create parameter class
    const plssvm::parameter params{ plssvm::kernel_type = static_cast<plssvm::kernel_function_type>(3) };

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid kernel type must throw
    EXPECT_THROW_WHAT(mock_csvm{ params },
                      plssvm::invalid_parameter_exception,
                      "Invalid kernel function 3 given!");
}
TEST(BaseCSVM, construct_from_parameter_invalid_gamma) {
    // create parameter class
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial, plssvm::gamma = -1.0 };

    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid value for gamma must throw
    EXPECT_THROW_WHAT(mock_csvm{ params },
                      plssvm::invalid_parameter_exception,
                      "gamma must be greater than 0.0, but is -1!");
}

TEST(BaseCSVM, construct_linear_from_named_parameters) {
    // correct parameters
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::cost = 2.0 };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ plssvm::kernel_type = params.kernel_type, plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}
TEST(BaseCSVM, construct_polynomial_from_named_parameters) {
    // correct parameters
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.1 }, plssvm::real_type{ 1.2 }, plssvm::real_type{ 0.001 } };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ plssvm::kernel_type = params.kernel_type,
                          plssvm::degree = params.degree,
                          plssvm::gamma = params.gamma,
                          plssvm::coef0 = params.coef0,
                          plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}
TEST(BaseCSVM, construct_rbf_from_named_parameters) {
    // correct parameters
    const plssvm::parameter params{ plssvm::kernel_type = plssvm::kernel_function_type::rbf, plssvm::gamma = 0.00001, plssvm::cost = 10.0 };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ plssvm::kernel_type = params.kernel_type,
                          plssvm::gamma = params.gamma,
                          plssvm::cost = params.cost };

    // check whether the parameters have been set correctly
    EXPECT_TRUE(csvm.get_params().equivalent(params));
}

TEST(BaseCSVM, construct_from_named_parameters_invalid_kernel_type) {
    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid kernel type must throw
    EXPECT_THROW_WHAT(mock_csvm{ plssvm::kernel_type = static_cast<plssvm::kernel_function_type>(3) },
                      plssvm::invalid_parameter_exception,
                      "Invalid kernel function 3 given!");
}
TEST(BaseCSVM, construct_from_named_parameters_invalid_gamma) {
    // creating a mock_csvm (since plssvm::csvm is pure virtual!) with an invalid value for gamma must throw
    EXPECT_THROW_WHAT((mock_csvm{ plssvm::kernel_type = plssvm::kernel_function_type::polynomial, plssvm::gamma = -1.0 }),
                      plssvm::invalid_parameter_exception,
                      "gamma must be greater than 0.0, but is -1!");
}

TEST(BaseCSVM, get_target_platforms) {
    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{};

    EXPECT_EQ(csvm.get_target_platform(), plssvm::target_platform::automatic);
}
TEST(BaseCSVM, get_params) {
    // create parameter class
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.01 } };

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ params };

    // check whether the parameters have been set correctly
    const plssvm::parameter csvm_params = csvm.get_params();
    EXPECT_EQ(csvm_params, params);
    EXPECT_TRUE(csvm_params.equivalent(params));
}

TEST(BaseCSVM, set_params_from_parameter) {
    // create mock_csvm (since plssvm::csvm is pure virtual!)
    mock_csvm csvm{};
    ASSERT_EQ(csvm.get_params(), plssvm::parameter{});

    // create parameter class
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.01 } };

    // set csvm parameter to new values
    csvm.set_params(params);

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}
TEST(BaseCSVM, set_params_from_named_parameters) {
    // create mock_csvm (since plssvm::csvm is pure virtual!)
    mock_csvm csvm{};
    ASSERT_EQ(csvm.get_params(), plssvm::parameter{});

    // create parameter class
    const plssvm::parameter params{ plssvm::kernel_function_type::polynomial, 4, plssvm::real_type{ 0.2 }, plssvm::real_type{ 0.1 }, plssvm::real_type{ 0.01 } };

    // set csvm parameter to new values
    csvm.set_params(plssvm::kernel_type = plssvm::kernel_function_type::polynomial,
                    plssvm::degree = 4,
                    plssvm::gamma = 0.2,
                    plssvm::coef0 = 0.1,
                    plssvm::cost = 0.01);

    // check whether the parameters have been set correctly
    EXPECT_EQ(csvm.get_params(), params);
}

TEST(BaseCSVM, csvm_backend_exists) {
    // test whether the given C-SVM backend exist
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::openmp::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::openmp::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::openmp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::openmp::csvm>::value);
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::cuda::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::cuda::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::cuda::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::cuda::csvm>::value);
#endif

#if defined(PLSSVM_HAS_HIP_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::hip::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::hip::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::hip::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::hip::csvm>::value);
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::opencl::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::opencl::csvm>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::opencl::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::opencl::csvm>::value);
#endif

#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::sycl::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::sycl::csvm>::value);
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvm>::value);
    #else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvm>::value);
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvm>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvm>::value);
    #else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvm>::value);
    #endif
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::sycl::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::sycl::csvm>::value);
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvm>::value);
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvm>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvm>::value);
#endif
}

class BaseCSVMWarning : public BaseCSVM, protected util::redirect_output<&std::clog> {};

TEST_F(BaseCSVMWarning, construct_unused_parameter_warning_degree) {
    // start capture of std::clog
    [[maybe_unused]] const mock_csvm csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::degree = 2 };
    // end capture of std::clog

    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("WARNING: degree parameter provided, which is not used in the linear kernel (u'*v)!"));
}
TEST_F(BaseCSVMWarning, construct_unused_parameter_warning_gamma) {
    // start capture of std::clog
    [[maybe_unused]] const mock_csvm csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::gamma = 0.1 };
    // end capture of std::clog

    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("WARNING: gamma parameter provided, which is not used in the linear kernel (u'*v)!"));
}
TEST_F(BaseCSVMWarning, construct_unused_parameter_warning_coef0) {
    // start capture of std::clog
    [[maybe_unused]] const mock_csvm csvm{ plssvm::kernel_type = plssvm::kernel_function_type::linear, plssvm::coef0 = 0.1 };
    // end capture of std::clog

    EXPECT_THAT(this->get_capture(), ::testing::HasSubstr("WARNING: coef0 parameter provided, which is not used in the linear kernel (u'*v)!"));
}

template <typename T>
class BaseCSVMMemberBase : public BaseCSVM, private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    static constexpr plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<0, T>;

    void SetUp() override {
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", data_set_file_.filename);
        const std::string model_template_file = fmt::format(PLSSVM_TEST_PATH "/data/model/{}_classes/6x4_linear_{}_TEMPLATE.libsvm.model",
                                                            util::get_num_classes<fixture_label_type>(),
                                                            fixture_classification);
        util::instantiate_template_file<fixture_label_type>(model_template_file, model_file_.filename);
    }

    /**
     * @brief Return the name of the instantiated data template file.
     * @return the file name (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::string &get_data_filename() const noexcept { return data_set_file_.filename; }
    /**
     * @brief Return the name of the instantiated model template file.
     * @return the file name (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::string &get_model_filename() const noexcept { return model_file_.filename; }

  private:
    /// The temporary data file.
    util::temporary_file data_set_file_{};
    /// The temporary model file.
    util::temporary_file model_file_{};
};

template <typename T>
class BaseCSVMFit : public BaseCSVM, private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    static constexpr plssvm::solver_type fixture_solver = util::test_parameter_value_at_v<0, T>;
    static constexpr plssvm::kernel_function_type fixture_kernel = util::test_parameter_value_at_v<1, T>;
    static constexpr plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<2, T>;

    void SetUp() override {
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/libsvm/6x4_TEMPLATE.libsvm", data_set_file_.filename);
    }

    /**
     * @brief Return the name of the instantiated data template file.
     * @return the file name (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::string &get_data_filename() const noexcept { return data_set_file_.filename; }

  private:
    /// The temporary data file.
    util::temporary_file data_set_file_{};
};
TYPED_TEST_SUITE(BaseCSVMFit, util::label_type_solver_and_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(BaseCSVMFit, fit) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));

    // mock the solve_lssvm_system_of_linear_equations function
    // clang-format off
    if constexpr (solver == plssvm::solver_type::automatic) {
        EXPECT_CALL(csvm, get_device_memory()).Times(num_calls);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvm, get_max_mem_alloc_size()).Times(num_calls);
#endif
    }
    EXPECT_CALL(csvm, setup_data_on_devices(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(num_calls)
                        .WillRepeatedly([]() { return plssvm::detail::simple_any{ util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(6, 4, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE) }; });
    EXPECT_CALL(csvm, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(num_calls)
                        .WillRepeatedly([]() { return plssvm::detail::simple_any{ util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(5, 5, plssvm::PADDING_SIZE, plssvm::PADDING_SIZE) }; });
    EXPECT_CALL(csvm, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(::testing::Between(num_calls * 1, num_calls * 6));  // at least once before CG loop, at most # data_points - 1 + 1
    // clang-format on

    // create data set
    const plssvm::data_set<label_type> training_data{ this->get_data_filename() };

    // call function
    const plssvm::model<label_type> model = csvm.fit(training_data, plssvm::solver = solver, plssvm::classification = classification);
    EXPECT_EQ(model.num_support_vectors(), 6);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.num_classes(), util::get_num_classes<label_type>());
    EXPECT_EQ(model.get_classification_type(), classification);
    EXPECT_EQ(model.get_params().kernel_type, kernel);
    EXPECT_EQ(model.get_params().gamma, 0.25);
}
TYPED_TEST(BaseCSVMFit, fit_named_parameters) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // skip unimplemented tests
    if constexpr (solver == plssvm::solver_type::cg_streaming || solver == plssvm::solver_type::cg_implicit) {
        GTEST_SKIP() << "Currently not implemented!";
    }

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));
    const int max_iter = 20;

    // mock the solve_lssvm_system_of_linear_equations function
    // clang-format off
    if constexpr (solver == plssvm::solver_type::automatic) {
        EXPECT_CALL(csvm, get_device_memory()).Times(num_calls);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvm, get_max_mem_alloc_size()).Times(num_calls);
#endif
    }
    EXPECT_CALL(csvm, setup_data_on_devices(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(num_calls)
                        .WillRepeatedly([]() { return plssvm::detail::simple_any{ util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(6, 4) }; });
    EXPECT_CALL(csvm, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(num_calls)
                        .WillRepeatedly([]() { return plssvm::detail::simple_any{ util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(5, 5) }; });
    EXPECT_CALL(csvm, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(::testing::Between(num_calls * 1, num_calls * (max_iter + 1)));  // at least once before CG loop, at most max_iter + 1 -> per classifier
    // clang-format on

    // create data set
    const plssvm::data_set<label_type> training_data{ this->get_data_filename() };

    // call function
    const plssvm::model<label_type> model = csvm.fit(training_data,
                                                     plssvm::solver = solver,
                                                     plssvm::classification = classification,
                                                     plssvm::epsilon = 1e-10,
                                                     plssvm::max_iter = max_iter);
    EXPECT_EQ(model.num_support_vectors(), 6);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.num_classes(), util::get_num_classes<label_type>());
    EXPECT_EQ(model.get_classification_type(), classification);
    EXPECT_EQ(model.get_params().kernel_type, kernel);
    EXPECT_EQ(model.get_params().gamma, 0.25);
}
TYPED_TEST(BaseCSVMFit, fit_named_parameters_invalid_epsilon) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // mock the solve_lssvm_system_of_linear_equations function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvm, get_device_memory()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvm, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvm, setup_data_on_devices(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    EXPECT_CALL(csvm, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvm, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create data set
    const plssvm::data_set<label_type> training_data{ this->get_data_filename() };

    // calling the function with an invalid epsilon should throw
    EXPECT_THROW_WHAT((std::ignore = csvm.fit(training_data, plssvm::solver = solver, plssvm::classification = classification, plssvm::epsilon = 0.0)),
                      plssvm::invalid_parameter_exception,
                      "epsilon must be less than 0.0, but is 0!");
}
TYPED_TEST(BaseCSVMFit, fit_named_parameters_invalid_max_iter) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // mock the solve_lssvm_system_of_linear_equations function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvm, get_device_memory()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvm, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvm, setup_data_on_devices(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    EXPECT_CALL(csvm, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvm, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create data set
    const plssvm::data_set<label_type> training_data{ this->get_data_filename() };

    // calling the function with an invalid epsilon should throw
    EXPECT_THROW_WHAT((std::ignore = csvm.fit(training_data, plssvm::solver = solver, plssvm::classification = classification, plssvm::max_iter = 0)),
                      plssvm::invalid_parameter_exception,
                      "max_iter must be greater than 0, but is 0!");
}
TYPED_TEST(BaseCSVMFit, fit_no_label) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // mock the solve_lssvm_system_of_linear_equations function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvm, get_device_memory()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvm, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvm, setup_data_on_devices(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    EXPECT_CALL(csvm, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvm, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const plssvm::detail::simple_any &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create data set without labels
    const plssvm::data_set<label_type> training_data{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };

    // in order to call fit, the provided data set must contain labels
    EXPECT_THROW_WHAT((std::ignore = csvm.fit(training_data, plssvm::solver = solver, plssvm::classification = classification)),
                      plssvm::invalid_parameter_exception,
                      "No labels given for training! Maybe the data is only usable for prediction?");
}

template <typename T>
class BaseCSVMPredict : public BaseCSVMMemberBase<T> {};
TYPED_TEST_SUITE(BaseCSVMPredict, util::label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(BaseCSVMPredict, predict) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{};

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));
    const int num_cols_in_return_matrix = classification == plssvm::classification_type::oaa ? static_cast<int>(util::get_num_classes<label_type>()) : 1;

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvm, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Return(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(6, num_cols_in_return_matrix)));
    // clang-format on

    // create data set and previously learned model
    const plssvm::data_set<label_type> data_to_predict{ this->get_data_filename() };
    const plssvm::model<label_type> learned_model{ this->get_model_filename() };

    // call function
    const std::vector<label_type> prediction = csvm.predict(learned_model, data_to_predict);
    EXPECT_EQ(prediction.size(), 6);
}
TYPED_TEST(BaseCSVMPredict, predict_num_feature_mismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvm, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set and previously learned model
    const plssvm::data_set<label_type> data_to_predict{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };
    const plssvm::model<label_type> learned_model{ this->get_model_filename() };

    // calling the function with mismatching number of features should throw
    EXPECT_THROW_WHAT(std::ignore = csvm.predict(learned_model, data_to_predict),
                      plssvm::invalid_parameter_exception,
                      "Number of features per data point (2) must match the number of features per support vector of the provided model (4)!");
}

template <typename T>
class BaseCSVMScore : public BaseCSVMMemberBase<T> {};
TYPED_TEST_SUITE(BaseCSVMScore, util::label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(BaseCSVMScore, score_model) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{};

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));
    const int num_cols_in_return_matrix = classification == plssvm::classification_type::oaa ? static_cast<int>(util::get_num_classes<label_type>()) : 1;

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvm, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Return(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(6, num_cols_in_return_matrix)));
    // clang-format on

    // read a previously learned model from a model file
    const plssvm::model<label_type> learned_model{ this->get_model_filename() };

    // call function
    const plssvm::real_type score = csvm.score(learned_model);
    EXPECT_INCLUSIVE_RANGE(score, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 });
}
TYPED_TEST(BaseCSVMScore, score_data_set) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{};

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));
    const int num_cols_in_return_matrix = classification == plssvm::classification_type::oaa ? static_cast<int>(util::get_num_classes<label_type>()) : 1;

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvm, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Return(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(6, num_cols_in_return_matrix)));
    // clang-format on

    // create data set and previously learned model
    const plssvm::data_set<label_type> data_to_score{ this->get_data_filename() };
    const plssvm::model<label_type> learned_model{ this->get_model_filename() };

    // call function
    const plssvm::real_type score = csvm.score(learned_model, data_to_score);
    EXPECT_INCLUSIVE_RANGE(score, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 });
}

TYPED_TEST(BaseCSVMScore, score_data_set_no_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvm, predict_values(
                        ::testing::An<const plssvm::parameter &>(),
                        ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                        ::testing::An<const std::vector<plssvm::real_type> &>(),
                        ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set
    const plssvm::data_set<label_type> data_to_score{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };
    // read a previously learned model from a model file
    const plssvm::model<label_type> learned_model{ this->get_model_filename() };

    // in order to call score, the provided data set must contain labels
    EXPECT_THROW_WHAT(std::ignore = csvm.score(learned_model, data_to_score), plssvm::invalid_parameter_exception, "The data set to score must have labels!");
}
TYPED_TEST(BaseCSVMScore, score_data_set_num_features_mismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create mock_csvm (since plssvm::csvm is pure virtual!)
    const mock_csvm csvm{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvm, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(labels.size(), 2);
    const plssvm::data_set<label_type> data_to_score{ data, labels };

    // read a previously learned model from a model file
    const plssvm::model<label_type> learned_model{ this->get_model_filename() };

    // calling the function with mismatching number of features should throw
    EXPECT_THROW_WHAT(std::ignore = csvm.score(learned_model, data_to_score),
                      plssvm::invalid_parameter_exception,
                      fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!",
                                  data.num_cols(),
                                  learned_model.num_features()));
}