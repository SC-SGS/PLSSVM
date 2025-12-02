/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the base C-SVC functions through its mock class.
 */

#include "plssvm/svm/csvc.hpp"  // plssvm::csvc

#include "plssvm/backend_types.hpp"                     // plssvm::csvm_backend_exists, plssvm::backend_csvm_type, plssvm::backend_csvm_type_t
#include "plssvm/backends/SYCL/detail/constants.hpp"    // NOLINT: namespace plssvm::sycl
#include "plssvm/classification_types.hpp"              // plssvm::classification_type
#include "plssvm/constants.hpp"                         // plssvm::real_type, plssvm::THREAD_BLOCK_SIZE, plssvm::INTERNAL_BLOCK_SIZE, plssvm::PADDING_SIZE
#include "plssvm/core.hpp"                              // NOLINT: include all csvm_backend_exists_v specializations
#include "plssvm/data_set/classification_data_set.hpp"  // plssvm::classification_data_set
#include "plssvm/detail/data_distribution.hpp"          // plssvm::detail::data_distribution::maximum_local_memory_needed
#include "plssvm/detail/memory_size.hpp"                // plssvm::detail::memory_size
#include "plssvm/detail/move_only_any.hpp"              // plssvm::detail::move_only_any
#include "plssvm/exceptions/exceptions.hpp"             // plssvm::invalid_parameter_exception
#include "plssvm/kernel_function_types.hpp"             // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                            // plssvm::aos_matrix
#include "plssvm/model/classification_model.hpp"        // plssvm::classification_model
#include "plssvm/parameter.hpp"                         // plssvm::parameter
#include "plssvm/solver_types.hpp"                      // plssvm::solver_type
#include "plssvm/svm/csvm.hpp"                          // plssvm::csvm_backend_exists_v

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_THROW_WHAT_MATCHER, EXPECT_INCLUSIVE_RANGE
#include "tests/naming.hpp"              // naming::parameter_definition_to_name
#include "tests/svm/mock_csvc.hpp"       // mock_csvc
#include "tests/types_to_test.hpp"       // util::classification_label_type_classification_type_gtest
#include "tests/utility.hpp"             // util::{redirect_output, temporary_file, instantiate_template_file, get_num_classes, calculate_number_of_classifiers,
                                         // generate_random_matrix, get_correct_data_file_labels}

#if defined(PLSSVM_HAS_MPI_ENABLED)
    #include "plssvm/mpi/communicator.hpp"  // plssvm::mpi::communicator

    #include "mpi.h"  // MPI_COMM_WORLD, MPI_Comm_dup, MPI_Comm_free
#endif

#include "fmt/format.h"   // fmt::format
#include "gmock/gmock.h"  // EXPECT_CALL, EXPECT_THAT, ::testing::{An, Between, Return, HasSubstr, ContainsRegex}
#include "gtest/gtest.h"  // TEST, TYPED_TEST, TYPED_TEST_SUITE, EXPECT_EQ, EXPECT_TRUE, EXPECT_FALSE, EXPECT_THAT,

#include <cstddef>   // std::size_t
#include <optional>  // std::optional, std::make_optional
#include <string>    // std::string
#include <tuple>     // std::ignore
#include <utility>   // std::move
#include <variant>   // std::holds_alternative
#include <vector>    // std::vector

class BaseCSVC : public ::testing::Test { };

TEST(BaseCSVC, CsvcBackendExists) {
    // test whether the given C-SVC backend exist
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::openmp::csvc>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::openmp::csvc>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::openmp::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::openmp::csvc>::value);
#endif

#if defined(PLSSVM_HAS_HPX_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::hpx::csvc>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::hpx::csvc>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::hpx::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::hpx::csvc>::value);
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::cuda::csvc>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::cuda::csvc>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::cuda::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::cuda::csvc>::value);
#endif

#if defined(PLSSVM_HAS_HIP_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::hip::csvc>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::hip::csvc>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::hip::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::hip::csvc>::value);
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::opencl::csvc>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::opencl::csvc>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::opencl::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::opencl::csvc>::value);
#endif

#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::sycl::csvc>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::sycl::csvc>::value);
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvc>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvc>::value);
    #else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvc>::value);
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvc>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvc>::value);
    #else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvc>::value);
    #endif
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::sycl::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::sycl::csvc>::value);
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvc>::value);
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvc>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvc>::value);
#endif
}

TEST(BaseCSVC, BackendCsvmType) {
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::openmp::backend_csvm_type<plssvm::csvc>::type, plssvm::openmp::csvc>();
    ::testing::StaticAssertTypeEq<plssvm::openmp::backend_csvm_type_t<plssvm::csvc>, plssvm::openmp::csvc>();
#endif

#if defined(PLSSVM_HAS_HPX_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::hpx::backend_csvm_type<plssvm::csvc>::type, plssvm::hpx::csvc>();
    ::testing::StaticAssertTypeEq<plssvm::hpx::backend_csvm_type_t<plssvm::csvc>, plssvm::hpx::csvc>();
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::cuda::backend_csvm_type<plssvm::csvc>::type, plssvm::cuda::csvc>();
    ::testing::StaticAssertTypeEq<plssvm::cuda::backend_csvm_type_t<plssvm::csvc>, plssvm::cuda::csvc>();
#endif

#if defined(PLSSVM_HAS_HIP_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::hip::backend_csvm_type<plssvm::csvc>::type, plssvm::hip::csvc>();
    ::testing::StaticAssertTypeEq<plssvm::hip::backend_csvm_type_t<plssvm::csvc>, plssvm::hip::csvc>();
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::opencl::backend_csvm_type<plssvm::csvc>::type, plssvm::opencl::csvc>();
    ::testing::StaticAssertTypeEq<plssvm::opencl::backend_csvm_type_t<plssvm::csvc>, plssvm::opencl::csvc>();
#endif

#if defined(PLSSVM_HAS_SYCL_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::sycl::backend_csvm_type<plssvm::csvc>::type, plssvm::sycl::csvc>();
    ::testing::StaticAssertTypeEq<plssvm::sycl::backend_csvm_type_t<plssvm::csvc>, plssvm::sycl::csvc>();
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    ::testing::StaticAssertTypeEq<plssvm::dpcpp::backend_csvm_type<plssvm::csvc>::type, plssvm::dpcpp::csvc>();
    ::testing::StaticAssertTypeEq<plssvm::dpcpp::backend_csvm_type_t<plssvm::csvc>, plssvm::dpcpp::csvc>();
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    ::testing::StaticAssertTypeEq<plssvm::adaptivecpp::backend_csvm_type<plssvm::csvc>::type, plssvm::adaptivecpp::csvc>();
    ::testing::StaticAssertTypeEq<plssvm::adaptivecpp::backend_csvm_type_t<plssvm::csvc>, plssvm::adaptivecpp::csvc>();
    #endif
#endif

#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::backend_csvm_type<plssvm::csvc>::type, plssvm::kokkos::csvc>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::backend_csvm_type_t<plssvm::csvc>, plssvm::kokkos::csvc>();
#endif
}

template <typename T>
class BaseCSVCMemberBase : public BaseCSVC,
                           private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<0, T>;

    void SetUp() override {
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", data_set_file_.filename);
        const std::string model_template_file = fmt::format(PLSSVM_TEST_PATH "/data/model/classification/6x4_{}_{}_TEMPLATE.libsvm.model",
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
    util::temporary_file data_set_file_;
    /// The temporary model file.
    util::temporary_file model_file_;
};

template <typename T>
class BaseCSVCFit : public BaseCSVC,
                    private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::solver_type fixture_solver = util::test_parameter_value_at_v<0, T>;
    constexpr static plssvm::kernel_function_type fixture_kernel = util::test_parameter_value_at_v<1, T>;
    constexpr static plssvm::classification_type fixture_classification = util::test_parameter_value_at_v<2, T>;

    void SetUp() override {
        util::instantiate_template_file<fixture_label_type>(PLSSVM_TEST_PATH "/data/libsvm/classification/6x4_TEMPLATE.libsvm", data_set_file_.filename);
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

TYPED_TEST_SUITE(BaseCSVCFit, util::classification_label_type_solver_and_kernel_function_and_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(BaseCSVCFit, Fit) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{ plssvm::parameter{ plssvm::kernel_type = kernel } };
    const std::size_t num_devices = csvc.num_available_devices();

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));

    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(num_calls);
    if constexpr (solver == plssvm::solver_type::automatic) {
        EXPECT_CALL(csvc, get_device_memory()).Times(num_calls);
        EXPECT_CALL(csvc, num_available_devices()).Times(num_calls);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvc, get_max_mem_alloc_size()).Times(num_calls);
#endif
    }
    EXPECT_CALL(csvc, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Invoke([num_devices]() {
                            std::vector<plssvm::detail::move_only_any> res(num_devices);
                            for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
                                auto matr = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 5, 5 }, plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE });
                                res[device_id] = plssvm::detail::move_only_any{ std::move(matr) };
                            }
                            return res; }));
    EXPECT_CALL(csvc, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(::testing::Between(num_calls * 1, num_calls * 6));  // at least once before CG loop, at most # data_points - 1 + 1
    // clang-format on

    // create data set
    plssvm::classification_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto &labels_opt = training_data.labels();
        if (labels_opt.has_value()) {
            training_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(training_data.data()), labels_opt.value() };
        }
    }

    // call function
    const plssvm::classification_model<label_type> model = csvc.fit(training_data, plssvm::solver = solver, plssvm::classification = classification);
    EXPECT_EQ(model.num_support_vectors(), 6);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.num_classes(), util::get_num_classes<label_type>());
    EXPECT_EQ(model.get_classification_type(), classification);
    EXPECT_EQ(model.get_params().kernel_type, kernel);
    ASSERT_TRUE(std::holds_alternative<plssvm::real_type>(model.get_params().gamma));
    EXPECT_EQ(std::get<plssvm::real_type>(model.get_params().gamma), plssvm::real_type{ 0.25 });
}

TYPED_TEST(BaseCSVCFit, FitNamedParameters) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{ plssvm::parameter{ plssvm::kernel_type = kernel } };
    const std::size_t num_devices = csvc.num_available_devices();

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));
    const int max_iter = 20;

    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(num_calls);
    if constexpr (solver == plssvm::solver_type::automatic) {
        EXPECT_CALL(csvc, get_device_memory()).Times(num_calls);
        EXPECT_CALL(csvc, num_available_devices()).Times(num_calls);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvc, get_max_mem_alloc_size()).Times(num_calls);
#endif
    }
    EXPECT_CALL(csvc, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Invoke([num_devices]() {
                            std::vector<plssvm::detail::move_only_any> res(num_devices);
                            for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
                                res[device_id] = plssvm::detail::move_only_any{ util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ 5, 5 },
                                                                                                                                                    plssvm::shape{ plssvm::PADDING_SIZE, plssvm::PADDING_SIZE }) };
                            }
                           return res; }));
    EXPECT_CALL(csvc, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(::testing::Between(num_calls * 1, num_calls * (max_iter + 1)));  // at least once before CG loop, at most max_iter + 1 -> per classifier
    // clang-format on

    // create data set
    plssvm::classification_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto &labels_opt = training_data.labels();
        if (labels_opt.has_value()) {
            training_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(training_data.data()), labels_opt.value() };
        }
    }

    // call function
    const plssvm::classification_model<label_type> model = csvc.fit(training_data,
                                                                    plssvm::solver = solver,
                                                                    plssvm::classification = classification,
                                                                    plssvm::epsilon = 1e-10,
                                                                    plssvm::max_iter = max_iter);
    EXPECT_EQ(model.num_support_vectors(), 6);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.num_classes(), util::get_num_classes<label_type>());
    EXPECT_EQ(model.get_classification_type(), classification);
    EXPECT_EQ(model.get_params().kernel_type, kernel);
    ASSERT_TRUE(std::holds_alternative<plssvm::real_type>(model.get_params().gamma));
    EXPECT_EQ(std::get<plssvm::real_type>(model.get_params().gamma), plssvm::real_type{ 0.25 });
}

TYPED_TEST(BaseCSVCFit, FitNamedParametersInvalidEpsilon) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create mock_csvc (since plssvm::csvc is pure virtual!)
    const mock_csvc csvc{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(0);
    EXPECT_CALL(csvc, get_device_memory()).Times(0);
    EXPECT_CALL(csvc, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvc, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvc, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvc, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create data set
    plssvm::classification_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto &labels_opt = training_data.labels();
        if (labels_opt.has_value()) {
            training_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(training_data.data()), labels_opt.value() };
        }
    }

    // calling the function with an invalid epsilon should throw
    EXPECT_THROW_WHAT((std::ignore = csvc.fit(training_data, plssvm::solver = solver, plssvm::classification = classification, plssvm::epsilon = 0.0)),
                      plssvm::invalid_parameter_exception,
                      "epsilon must be less than 0.0, but is 0!");
}

TYPED_TEST(BaseCSVCFit, FitNamedParametersInvalidMaxIter) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(0);
    EXPECT_CALL(csvc, get_device_memory()).Times(0);
    EXPECT_CALL(csvc, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvc, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvc, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvc, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create data set
    plssvm::classification_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto &labels_opt = training_data.labels();
        if (labels_opt.has_value()) {
            training_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(training_data.data()), labels_opt.value() };
        }
    }

    // calling the function with an invalid epsilon should throw
    EXPECT_THROW_WHAT((std::ignore = csvc.fit(training_data, plssvm::solver = solver, plssvm::classification = classification, plssvm::max_iter = 0)),
                      plssvm::invalid_parameter_exception,
                      "max_iter must be greater than 0, but is 0!");
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(BaseCSVCFit, FitCommunicatorMismatch) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(0);
    EXPECT_CALL(csvc, get_device_memory()).Times(0);
    EXPECT_CALL(csvc, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvc, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvc, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvc, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create mismatching MPI communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data set
    plssvm::classification_data_set<label_type> training_data{ comm, this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto &labels_opt = training_data.labels();
        if (labels_opt.has_value()) {
            training_data = plssvm::classification_data_set<label_type>{ comm, util::matrix_abs(training_data.data()), labels_opt.value() };
        }
    }

    // calling the function with mismatching MPI communicators should throw
    EXPECT_THROW_WHAT((std::ignore = csvc.fit(training_data, plssvm::solver = solver, plssvm::classification = classification)),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVC and data set must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(BaseCSVCFit, FitNoLabel) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(0);
    EXPECT_CALL(csvc, get_device_memory()).Times(0);
    EXPECT_CALL(csvc, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvc, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvc, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvc, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create data set without labels
    plssvm::classification_data_set<label_type> training_data{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        training_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(training_data.data()) };
    }

    // in order to call fit, the provided data set must contain labels
    EXPECT_THROW_WHAT((std::ignore = csvc.fit(training_data, plssvm::solver = solver, plssvm::classification = classification)),
                      plssvm::invalid_parameter_exception,
                      "No labels given for training! Maybe the data is only usable for prediction?");
}

TYPED_TEST(BaseCSVCFit, FitOutOfResources) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // this test is only really applicable for the automatic solver type
    if constexpr (solver == plssvm::solver_type::automatic) {
        // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
        const mock_csvc csvc{ plssvm::parameter{ plssvm::kernel_type = kernel } };

        // override on call
        using namespace plssvm::detail::literals;  // NOLINT(google-build-using-namespace): only imports custom user-defined literals into this namespace
        ON_CALL(csvc, get_device_memory()).WillByDefault(::testing::Return(std::vector<plssvm::detail::memory_size>{ 512_MiB + 1_KiB, 512_MiB + 1_KiB }));

        // clang-format off
        EXPECT_CALL(csvc, get_local_memory()).Times(1);
        EXPECT_CALL(csvc, get_device_memory()).Times(1);
        EXPECT_CALL(csvc, num_available_devices()).Times(1);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvc, get_max_mem_alloc_size()).Times(1);
#endif
        EXPECT_CALL(csvc, assemble_kernel_matrix(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<const plssvm::parameter &>(),
                                ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                                ::testing::An<const std::vector<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>()))
                            .Times(0);
        EXPECT_CALL(csvc, blas_level_3(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                                ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                            .Times(0);
        // clang-format on

        // create data set
        plssvm::classification_data_set<label_type> training_data{ this->get_data_filename() };
        if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
            // chi-squared is well-defined for non-negative values only
            const auto &labels_opt = training_data.labels();
            if (labels_opt.has_value()) {
                training_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(training_data.data()), labels_opt.value() };
            }
        }

        // call function -> should throw since we are out of resources
        EXPECT_THROW_WHAT_MATCHER((std::ignore = csvc.fit(training_data, plssvm::solver = solver, plssvm::classification = classification)),
                                  plssvm::kernel_launch_resources,
                                  ::testing::ContainsRegex("Not enough device memory available on device.* even for the cg_implicit solver!"));
    }
}

TYPED_TEST(BaseCSVCFit, FitDeviceMemoryTooSmall) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // this test is only really applicable for the automatic solver type
    if constexpr (solver == plssvm::solver_type::automatic) {
        // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
        const mock_csvc csvc{ plssvm::parameter{ plssvm::kernel_type = kernel } };

        // override on call
        using namespace plssvm::detail::literals;  // NOLINT(google-build-using-namespace): only imports custom user-defined literals into this namespace
        ON_CALL(csvc, get_device_memory()).WillByDefault(::testing::Return(std::vector<plssvm::detail::memory_size>{ 1_KiB, 1_KiB }));

        // clang-format off
        EXPECT_CALL(csvc, get_local_memory()).Times(1);
        EXPECT_CALL(csvc, get_device_memory()).Times(1);
        EXPECT_CALL(csvc, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvc, get_max_mem_alloc_size()).Times(0);
#endif
        EXPECT_CALL(csvc, assemble_kernel_matrix(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<const plssvm::parameter &>(),
                                ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                                ::testing::An<const std::vector<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>()))
                            .Times(0);
        EXPECT_CALL(csvc, blas_level_3(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                                ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                            .Times(0);
        // clang-format on

        // create data set
        plssvm::classification_data_set<label_type> training_data{ this->get_data_filename() };
        if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
            // chi-squared is well-defined for non-negative values only
            const auto &labels_opt = training_data.labels();
            if (labels_opt.has_value()) {
                training_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(training_data.data()), labels_opt.value() };
            }
        }

        // call function -> should throw since we are out of resources
        EXPECT_THROW_WHAT((std::ignore = csvc.fit(training_data, plssvm::solver = solver, plssvm::classification = classification)),
                          plssvm::kernel_launch_resources,
                          "At least 512.00 MiB of memory must be available, but available are only 1.00 KiB!");
    }
}

TYPED_TEST(BaseCSVCFit, FitLocalMemoryTooSmall) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // override on call
    constexpr plssvm::detail::memory_size needed_local_mem_size = plssvm::detail::data_distribution::maximum_local_memory_needed();
    ON_CALL(csvc, get_local_memory()).WillByDefault(::testing::Return((std::vector<std::optional<plssvm::detail::memory_size>>{ std::make_optional(needed_local_mem_size / 2), std::make_optional(needed_local_mem_size / 2) })));

    EXPECT_CALL(csvc, get_local_memory()).Times(1);
    // this test is only really applicable for the automatic solver type
    if constexpr (solver == plssvm::solver_type::automatic) {
        // clang-format off
        EXPECT_CALL(csvc, get_device_memory()).Times(0);
        EXPECT_CALL(csvc, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvc, get_max_mem_alloc_size()).Times(0);
#endif
        EXPECT_CALL(csvc, assemble_kernel_matrix(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<const plssvm::parameter &>(),
                                ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                                ::testing::An<const std::vector<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>()))
                            .Times(0);
        EXPECT_CALL(csvc, blas_level_3(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                                ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>()))
                            .Times(0);
        // clang-format on
    }

    // create data set
    plssvm::classification_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        const auto &labels_opt = training_data.labels();
        if (labels_opt.has_value()) {
            training_data = plssvm::classification_data_set<label_type>{ util::matrix_abs(training_data.data()), labels_opt.value() };
        }
    }

    // call function -> should throw since we are out of resources
    EXPECT_THROW_WHAT((std::ignore = csvc.fit(training_data, plssvm::solver = solver, plssvm::classification = classification)),
                      plssvm::kernel_launch_resources,
                      fmt::format("At least {} of local memory must be available for the hyperparameter combination THREAD_BLOCK_SIZE={} and INTERNAL_BLOCK_SIZE={}, but available are only {}!",
                                  needed_local_mem_size,
                                  plssvm::THREAD_BLOCK_SIZE,
                                  plssvm::INTERNAL_BLOCK_SIZE,
                                  needed_local_mem_size / 2));
}

template <typename T>
class BaseCSVCPredict : public BaseCSVCMemberBase<T> { };

TYPED_TEST_SUITE(BaseCSVCPredict, util::classification_label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(BaseCSVCPredict, Predict) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{};

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));
    const std::size_t num_cols_in_return_matrix = classification == plssvm::classification_type::oaa ? util::get_num_classes<label_type>() : std::size_t{ 1 };

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(1);
    EXPECT_CALL(csvc, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Return(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, num_cols_in_return_matrix })));
    // clang-format on

    // create data set and previously learned model
    const plssvm::classification_data_set<label_type> data_to_predict{ this->get_data_filename() };
    const plssvm::classification_model<label_type> learned_model{ this->get_model_filename() };

    // call function
    const std::vector<label_type> prediction = csvc.predict(learned_model, data_to_predict);
    EXPECT_EQ(prediction.size(), 6);
}

TYPED_TEST(BaseCSVCPredict, PredictNumFeatureMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(0);
    EXPECT_CALL(csvc, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set and previously learned model
    const plssvm::classification_data_set<label_type> data_to_predict{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };
    const plssvm::classification_model<label_type> learned_model{ this->get_model_filename() };

    // calling the function with mismatching number of features should throw
    EXPECT_THROW_WHAT(std::ignore = csvc.predict(learned_model, data_to_predict),
                      plssvm::invalid_parameter_exception,
                      "Number of features per data point (2) must match the number of features per support vector of the provided model (4)!");
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(BaseCSVCPredict, PredictCommismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(0);
    EXPECT_CALL(csvc, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create mismatching MPI communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data set and previously learned model
    const plssvm::classification_data_set<label_type> data_to_predict{ this->get_data_filename() };
    const plssvm::classification_data_set<label_type> data_to_predict_wrong_comm{ comm, this->get_data_filename() };
    const plssvm::classification_model<label_type> learned_model{ this->get_model_filename() };
    const plssvm::classification_model<label_type> learned_model_wrong_comm{ comm, this->get_model_filename() };

    // calling the function with mismatching MPI communicators should throw
    EXPECT_THROW_WHAT(std::ignore = csvc.predict(learned_model_wrong_comm, data_to_predict),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVC and model must be identical!");
    EXPECT_THROW_WHAT(std::ignore = csvc.predict(learned_model, data_to_predict_wrong_comm),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVC and data set must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

template <typename T>
class BaseCSVCScore : public BaseCSVCMemberBase<T> { };

TYPED_TEST_SUITE(BaseCSVCScore, util::classification_label_type_classification_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(BaseCSVCScore, ScoreModel) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{};

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));
    const std::size_t num_cols_in_return_matrix = classification == plssvm::classification_type::oaa ? util::get_num_classes<label_type>() : std::size_t{ 1 };

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(1);
    EXPECT_CALL(csvc, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Return(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, num_cols_in_return_matrix })));
    // clang-format on

    // read a previously learned model from a model file
    const plssvm::classification_model<label_type> learned_model{ this->get_model_filename() };

    // call function
    const plssvm::real_type score = csvc.score(learned_model);
    EXPECT_INCLUSIVE_RANGE(score, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 });
}

TYPED_TEST(BaseCSVCScore, ScoreDataSet) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::classification_type classification = TestFixture::fixture_classification;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{};

    // determine the EXPECT_CALL values for the current classification type
    const int num_calls = classification == plssvm::classification_type::oaa ? 1 : static_cast<int>(util::calculate_number_of_classifiers(plssvm::classification_type::oao, util::get_num_classes<label_type>()));
    const std::size_t num_cols_in_return_matrix = classification == plssvm::classification_type::oaa ? util::get_num_classes<label_type>() : std::size_t{ 1 };

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(1);
    EXPECT_CALL(csvc, predict_values(
                        ::testing::An<const plssvm::parameter &>(),
                        ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                        ::testing::An<const std::vector<plssvm::real_type> &>(),
                        ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>()))
                    .Times(num_calls)
                    .WillRepeatedly(::testing::Return(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, num_cols_in_return_matrix })));
    // clang-format on

    // create data set and previously learned model
    const plssvm::classification_data_set<label_type> data_to_score{ this->get_data_filename() };
    const plssvm::classification_model<label_type> learned_model{ this->get_model_filename() };

    // call function
    const plssvm::real_type score = csvc.score(learned_model, data_to_score);
    EXPECT_INCLUSIVE_RANGE(score, plssvm::real_type{ 0.0 }, plssvm::real_type{ 1.0 });
}

TYPED_TEST(BaseCSVCScore, ScoreDataSetNoLabel) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(0);
    EXPECT_CALL(csvc, predict_values(
                    ::testing::An<const plssvm::parameter &>(),
                    ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                    ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                    ::testing::An<const std::vector<plssvm::real_type> &>(),
                    ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                    ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set
    const plssvm::classification_data_set<label_type> data_to_score{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };
    // read a previously learned model from a model file
    const plssvm::classification_model<label_type> learned_model{ this->get_model_filename() };

    // in order to call score, the provided data set must contain labels
    EXPECT_THROW_WHAT(std::ignore = csvc.score(learned_model, data_to_score), plssvm::invalid_parameter_exception, "The data set to score must have labels!");
}

TYPED_TEST(BaseCSVCScore, ScoreDataSetNumFeaturesMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(0);
    EXPECT_CALL(csvc, predict_values(
                        ::testing::An<const plssvm::parameter &>(),
                        ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                        ::testing::An<const std::vector<plssvm::real_type> &>(),
                        ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto data = util::generate_random_matrix<plssvm::soa_matrix<plssvm::real_type>>(plssvm::shape{ labels.size(), 2 });
    const plssvm::classification_data_set<label_type> data_to_score{ data, labels };

    // read a previously learned model from a model file
    const plssvm::classification_model<label_type> learned_model{ this->get_model_filename() };

    // calling the function with mismatching number of features should throw
    EXPECT_THROW_WHAT(std::ignore = csvc.score(learned_model, data_to_score),
                      plssvm::invalid_parameter_exception,
                      fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!",
                                  data.num_cols(),
                                  learned_model.num_features()));
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(BaseCSVCScore, PredictCommMismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(0);
    EXPECT_CALL(csvc, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create mismatching MPI communicator
    MPI_Comm duplicated_mpi_comm{};
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data set and previously learned model
    const plssvm::classification_data_set<label_type> data_to_predict{ this->get_data_filename() };
    const plssvm::classification_data_set<label_type> data_to_predict_wrong_comm{ comm, this->get_data_filename() };
    const plssvm::classification_model<label_type> learned_model{ this->get_model_filename() };
    const plssvm::classification_model<label_type> learned_model_wrong_comm{ comm, this->get_model_filename() };

    // calling the function with mismatching MPI communicators should throw
    EXPECT_THROW_WHAT(std::ignore = csvc.score(learned_model_wrong_comm, data_to_predict),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVC and model must be identical!");
    EXPECT_THROW_WHAT(std::ignore = csvc.score(learned_model, data_to_predict_wrong_comm),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVC and data set must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(BaseCSVCScore, PredictLocalMemoryTooSmall) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVC: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvc csvc{};

    // override on call
    constexpr plssvm::detail::memory_size needed_local_mem_size = plssvm::detail::data_distribution::maximum_local_memory_needed();
    ON_CALL(csvc, get_local_memory()).WillByDefault(::testing::Return((std::vector<std::optional<plssvm::detail::memory_size>>{ std::make_optional(needed_local_mem_size / 2), std::make_optional(needed_local_mem_size / 2) })));

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvc, get_local_memory()).Times(1);
    EXPECT_CALL(csvc, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::soa_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::soa_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set and previously learned model
    const plssvm::classification_data_set<label_type> data_to_predict{ this->get_data_filename() };
    const plssvm::classification_model<label_type> learned_model{ this->get_model_filename() };

    // calling the function with mismatching MPI communicators should throw
    EXPECT_THROW_WHAT(std::ignore = csvc.score(learned_model, data_to_predict),
                      plssvm::kernel_launch_resources,
                      fmt::format("At least {} of local memory must be available for the hyperparameter combination THREAD_BLOCK_SIZE={} and INTERNAL_BLOCK_SIZE={}, but available are only {}!",
                                  needed_local_mem_size,
                                  plssvm::THREAD_BLOCK_SIZE,
                                  plssvm::INTERNAL_BLOCK_SIZE,
                                  needed_local_mem_size / 2));
}
