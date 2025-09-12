/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Tests for the base C-SVR functions through its mock class.
 */

#include "plssvm/svm/csvr.hpp"  // plssvm::csvr

#include "plssvm/backend_types.hpp"                 // plssvm::csvm_backend_exists, plssvm::csvm_backend_exists_v, plssvm::backend_csvm_type, plssvm::backend_csvm_type_t
#include "plssvm/constants.hpp"                     // plssvm::real_type
#include "plssvm/core.hpp"                          // sycl namespace handling
#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set
#include "plssvm/detail/data_distribution.hpp"      // plssvm::detail::data_distribution::maximum_local_memory_needed
#include "plssvm/detail/memory_size.hpp"            // plssvm::detail::memory_size
#include "plssvm/detail/move_only_any.hpp"          // plssvm::detail::move_only_any
#include "plssvm/exceptions/exceptions.hpp"         // plssvm::invalid_parameter_exception
#include "plssvm/kernel_function_types.hpp"         // plssvm::kernel_function_type
#include "plssvm/matrix.hpp"                        // plssvm::aos_matrix
#include "plssvm/model/regression_model.hpp"        // plssvm::regression_model
#include "plssvm/parameter.hpp"                     // plssvm::parameter
#include "plssvm/solver_types.hpp"                  // plssvm::solver_type

#include "tests/custom_test_macros.hpp"  // EXPECT_THROW_WHAT, EXPECT_THROW_WHAT_MATCHER, EXPECT_INCLUSIVE_RANGE
#include "tests/naming.hpp"              // naming::parameter_definition_to_name
#include "tests/svm/mock_csvr.hpp"       // mock_csvr
#include "tests/types_to_test.hpp"       // util::regression_label_type_classification_type_gtest
#include "tests/utility.hpp"             // util::{redirect_output, temporary_file, instantiate_template_file, generate_random_matrix, get_correct_data_file_labels}

#if defined(PLSSVM_HAS_MPI_ENABLED)
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

class BaseCSVR : public ::testing::Test { };

TEST(BaseCSVR, csvr_backend_exists) {
    // test whether the given C-SVR backend exist
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::openmp::csvr>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::openmp::csvr>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::openmp::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::openmp::csvr>::value);
#endif

#if defined(PLSSVM_HAS_HPX_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::hpx::csvr>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::hpx::csvr>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::hpx::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::hpx::csvr>::value);
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::cuda::csvr>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::cuda::csvr>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::cuda::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::cuda::csvr>::value);
#endif

#if defined(PLSSVM_HAS_HIP_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::hip::csvr>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::hip::csvr>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::hip::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::hip::csvr>::value);
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::opencl::csvr>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::opencl::csvr>::value);
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::opencl::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::opencl::csvr>::value);
#endif

#if defined(PLSSVM_HAS_SYCL_BACKEND)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::sycl::csvr>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::sycl::csvr>::value);
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvr>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvr>::value);
    #else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvr>::value);
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    EXPECT_TRUE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvr>);
    EXPECT_TRUE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvr>::value);
    #else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvr>::value);
    #endif
#else
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::sycl::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::sycl::csvr>::value);
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::dpcpp::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::dpcpp::csvr>::value);
    EXPECT_FALSE(plssvm::csvm_backend_exists_v<plssvm::adaptivecpp::csvr>);
    EXPECT_FALSE(plssvm::csvm_backend_exists<plssvm::adaptivecpp::csvr>::value);
#endif
}

TEST(BaseCSVR, backend_csvm_type) {
#if defined(PLSSVM_HAS_OPENMP_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::openmp::backend_csvm_type<plssvm::csvr>::type, plssvm::openmp::csvr>();
    ::testing::StaticAssertTypeEq<plssvm::openmp::backend_csvm_type_t<plssvm::csvr>, plssvm::openmp::csvr>();
#endif

#if defined(PLSSVM_HAS_HPX_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::hpx::backend_csvm_type<plssvm::csvr>::type, plssvm::hpx::csvr>();
    ::testing::StaticAssertTypeEq<plssvm::hpx::backend_csvm_type_t<plssvm::csvr>, plssvm::hpx::csvr>();
#endif

#if defined(PLSSVM_HAS_CUDA_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::cuda::backend_csvm_type<plssvm::csvr>::type, plssvm::cuda::csvr>();
    ::testing::StaticAssertTypeEq<plssvm::cuda::backend_csvm_type_t<plssvm::csvr>, plssvm::cuda::csvr>();
#endif

#if defined(PLSSVM_HAS_HIP_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::hip::backend_csvm_type<plssvm::csvr>::type, plssvm::hip::csvr>();
    ::testing::StaticAssertTypeEq<plssvm::hip::backend_csvm_type_t<plssvm::csvr>, plssvm::hip::csvr>();
#endif

#if defined(PLSSVM_HAS_OPENCL_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::opencl::backend_csvm_type<plssvm::csvr>::type, plssvm::opencl::csvr>();
    ::testing::StaticAssertTypeEq<plssvm::opencl::backend_csvm_type_t<plssvm::csvr>, plssvm::opencl::csvr>();
#endif

#if defined(PLSSVM_HAS_SYCL_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::sycl::backend_csvm_type<plssvm::csvr>::type, plssvm::sycl::csvr>();
    ::testing::StaticAssertTypeEq<plssvm::sycl::backend_csvm_type_t<plssvm::csvr>, plssvm::sycl::csvr>();
    #if defined(PLSSVM_SYCL_BACKEND_HAS_DPCPP)
    ::testing::StaticAssertTypeEq<plssvm::dpcpp::backend_csvm_type<plssvm::csvr>::type, plssvm::dpcpp::csvr>();
    ::testing::StaticAssertTypeEq<plssvm::dpcpp::backend_csvm_type_t<plssvm::csvr>, plssvm::dpcpp::csvr>();
    #endif
    #if defined(PLSSVM_SYCL_BACKEND_HAS_ADAPTIVECPP)
    ::testing::StaticAssertTypeEq<plssvm::adaptivecpp::backend_csvm_type<plssvm::csvr>::type, plssvm::adaptivecpp::csvr>();
    ::testing::StaticAssertTypeEq<plssvm::adaptivecpp::backend_csvm_type_t<plssvm::csvr>, plssvm::adaptivecpp::csvr>();
    #endif
#endif

#if defined(PLSSVM_HAS_KOKKOS_BACKEND)
    ::testing::StaticAssertTypeEq<plssvm::kokkos::backend_csvm_type<plssvm::csvr>::type, plssvm::kokkos::csvr>();
    ::testing::StaticAssertTypeEq<plssvm::kokkos::backend_csvm_type_t<plssvm::csvr>, plssvm::kokkos::csvr>();
#endif
}

template <typename T>
class BaseCSVRMemberBase : public BaseCSVR,
                           private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;

    void SetUp() override {
        const std::string model_template_file = PLSSVM_TEST_PATH "/data/model/regression/6x4_TEMPLATE.libsvm.model";
        util::instantiate_template_file<fixture_label_type>(model_template_file, model_file_.filename);
    }

    /**
     * @brief Return the name of the instantiated data template file.
     * @return the file name (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::string &get_data_filename() const noexcept { return data_set_file_name_; }

    /**
     * @brief Return the name of the instantiated model template file.
     * @return the file name (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::string &get_model_filename() const noexcept { return model_file_.filename; }

  private:
    /// The name of the data set file.
    std::string data_set_file_name_{ PLSSVM_TEST_PATH "/data/libsvm/regression/6x4.libsvm" };
    /// The temporary model file.
    util::temporary_file model_file_{};
};

template <typename T>
class BaseCSVRFit : public BaseCSVR,
                    private util::redirect_output<> {
  protected:
    using fixture_label_type = util::test_parameter_type_at_t<0, T>;
    constexpr static plssvm::solver_type fixture_solver = util::test_parameter_value_at_v<0, T>;
    constexpr static plssvm::kernel_function_type fixture_kernel = util::test_parameter_value_at_v<1, T>;

    /**
     * @brief Return the name of the instantiated data template file.
     * @return the file name (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::string &get_data_filename() const noexcept { return data_set_file_name_; }

  private:
    /// The name of the data set file.
    std::string data_set_file_name_{ PLSSVM_TEST_PATH "/data/libsvm/regression/6x4.libsvm" };
};

TYPED_TEST_SUITE(BaseCSVRFit, util::regression_label_type_solver_and_kernel_function_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(BaseCSVRFit, fit) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{ plssvm::parameter{ plssvm::kernel_type = kernel } };
    const std::size_t num_devices = csvr.num_available_devices();

    // for the C-SVR, every function will be called exactly once
    const int num_calls = 1;

    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(num_calls);
    if constexpr (solver == plssvm::solver_type::automatic) {
        EXPECT_CALL(csvr, get_device_memory()).Times(num_calls);
        EXPECT_CALL(csvr, num_available_devices()).Times(num_calls);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvr, get_max_mem_alloc_size()).Times(num_calls);
#endif
    }
    EXPECT_CALL(csvr, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Invoke([num_devices]() {
                            std::vector<plssvm::detail::move_only_any> res(num_devices);
                            for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
                                auto matr = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 5, 5 });
                                res[device_id] = plssvm::detail::move_only_any{ std::move(matr) };
                            }
                            return res; }));
    EXPECT_CALL(csvr, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>()))
                        .Times(::testing::Between(num_calls * 1, num_calls * 6));  // at least once before CG loop, at most # data_points - 1 + 1
    // clang-format on

    // create data set
    plssvm::regression_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (training_data.labels().has_value()) {
            training_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(training_data.data()), *training_data.labels() };
        }
    }

    // call function
    const plssvm::regression_model<label_type> model = csvr.fit(training_data, plssvm::solver = solver);
    EXPECT_EQ(model.num_support_vectors(), 6);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.get_params().kernel_type, kernel);
    ASSERT_TRUE(std::holds_alternative<plssvm::real_type>(model.get_params().gamma));
    EXPECT_EQ(std::get<plssvm::real_type>(model.get_params().gamma), plssvm::real_type{ 0.25 });
}

TYPED_TEST(BaseCSVRFit, fit_named_parameters) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{ plssvm::parameter{ plssvm::kernel_type = kernel } };
    const std::size_t num_devices = csvr.num_available_devices();

    // for the C-SVR, every function will be called exactly once
    const int num_calls = 1;
    const int max_iter = 20;

    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(num_calls);
    if constexpr (solver == plssvm::solver_type::automatic) {
        EXPECT_CALL(csvr, get_device_memory()).Times(num_calls);
        EXPECT_CALL(csvr, num_available_devices()).Times(num_calls);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvr, get_max_mem_alloc_size()).Times(num_calls);
#endif
    }
    EXPECT_CALL(csvr, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Invoke([num_devices]() {
                            std::vector<plssvm::detail::move_only_any> res(num_devices);
                            for (std::size_t device_id = 0; device_id < num_devices; ++device_id) {
                                res[device_id] = plssvm::detail::move_only_any{ util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 5, 5 }) };
                            }
                           return res; }));
    EXPECT_CALL(csvr, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>()))
                        .Times(::testing::Between(num_calls * 1, num_calls * (max_iter + 1)));  // at least once before CG loop, at most max_iter + 1 -> per classifier
    // clang-format on

    // create data set
    plssvm::regression_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (training_data.labels().has_value()) {
            training_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(training_data.data()), *training_data.labels() };
        }
    }

    // call function
    const plssvm::regression_model<label_type> model = csvr.fit(training_data,
                                                                plssvm::solver = solver,
                                                                plssvm::epsilon = 1e-10,
                                                                plssvm::max_iter = max_iter);
    EXPECT_EQ(model.num_support_vectors(), 6);
    EXPECT_EQ(model.num_features(), 4);
    EXPECT_EQ(model.get_params().kernel_type, kernel);
    ASSERT_TRUE(std::holds_alternative<plssvm::real_type>(model.get_params().gamma));
    EXPECT_EQ(std::get<plssvm::real_type>(model.get_params().gamma), plssvm::real_type{ 0.25 });
}

TYPED_TEST(BaseCSVRFit, fit_named_parameters_invalid_epsilon) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // create mock_csvr (since plssvm::csvr is pure virtual!)
    const mock_csvr csvr{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, get_device_memory()).Times(0);
    EXPECT_CALL(csvr, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvr, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvr, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvr, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create data set
    plssvm::regression_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (training_data.labels().has_value()) {
            training_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(training_data.data()), *training_data.labels() };
        }
    }

    // calling the function with an invalid epsilon should throw
    EXPECT_THROW_WHAT((std::ignore = csvr.fit(training_data, plssvm::solver = solver, plssvm::epsilon = 0.0)),
                      plssvm::invalid_parameter_exception,
                      "epsilon must be less than 0.0, but is 0!");
}

TYPED_TEST(BaseCSVRFit, fit_named_parameters_invalid_max_iter) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, get_device_memory()).Times(0);
    EXPECT_CALL(csvr, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvr, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvr, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvr, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create data set
    plssvm::regression_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (training_data.labels().has_value()) {
            training_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(training_data.data()), *training_data.labels() };
        }
    }

    // calling the function with an invalid epsilon should throw
    EXPECT_THROW_WHAT((std::ignore = csvr.fit(training_data, plssvm::solver = solver, plssvm::max_iter = 0)),
                      plssvm::invalid_parameter_exception,
                      "max_iter must be greater than 0, but is 0!");
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(BaseCSVRFit, fit_communicator_mismatch) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, get_device_memory()).Times(0);
    EXPECT_CALL(csvr, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvr, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvr, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvr, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create mismatching MPI communicator
    MPI_Comm duplicated_mpi_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data set
    plssvm::regression_data_set<label_type> training_data{ comm, this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (training_data.labels().has_value()) {
            training_data = plssvm::regression_data_set<label_type>{ comm, util::matrix_abs(training_data.data()), *training_data.labels() };
        }
    }

    // calling the function with mismatching MPI communicators should throw
    EXPECT_THROW_WHAT((std::ignore = csvr.fit(training_data, plssvm::solver = solver)),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVR and data set must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(BaseCSVRFit, fit_no_label) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, get_device_memory()).Times(0);
    EXPECT_CALL(csvr, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
    EXPECT_CALL(csvr, get_max_mem_alloc_size()).Times(0);
#endif
    EXPECT_CALL(csvr, assemble_kernel_matrix(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>()))
                        .Times(0);
    EXPECT_CALL(csvr, blas_level_3(
                            ::testing::An<plssvm::solver_type>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<plssvm::real_type>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>()))
                        .Times(0);
    // clang-format on

    // create data set without labels
    plssvm::regression_data_set<label_type> training_data{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        training_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(training_data.data()) };
    }

    // in order to call fit, the provided data set must contain labels
    EXPECT_THROW_WHAT((std::ignore = csvr.fit(training_data, plssvm::solver = solver)),
                      plssvm::invalid_parameter_exception,
                      "No labels given for training! Maybe the data is only usable for prediction?");
}

TYPED_TEST(BaseCSVRFit, fit_out_of_resources) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // this test is only really applicable for the automatic solver type
    if constexpr (solver == plssvm::solver_type::automatic) {
        // create C-SVC: must be done using the mock class since the csvr base class is pure virtual
        const mock_csvr csvr{ plssvm::parameter{ plssvm::kernel_type = kernel } };

        // override on call
        using namespace plssvm::detail::literals;
        ON_CALL(csvr, get_device_memory()).WillByDefault(::testing::Return(std::vector<plssvm::detail::memory_size>{ 512_MiB + 1_KiB, 512_MiB + 1_KiB }));

        // clang-format off
        EXPECT_CALL(csvr, get_local_memory()).Times(1);
        EXPECT_CALL(csvr, get_device_memory()).Times(1);
        EXPECT_CALL(csvr, num_available_devices()).Times(1);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvr, get_max_mem_alloc_size()).Times(1);
#endif
        EXPECT_CALL(csvr, assemble_kernel_matrix(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<const plssvm::parameter &>(),
                                ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                                ::testing::An<const std::vector<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>()))
                            .Times(0);
        EXPECT_CALL(csvr, blas_level_3(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                                ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>()))
                            .Times(0);
        // clang-format on

        // create data set
        plssvm::regression_data_set<label_type> training_data{ PLSSVM_REGRESSION_TEST_FILE };
        if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
            // chi-squared is well-defined for non-negative values only
            if (training_data.labels().has_value()) {
                training_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(training_data.data()), *training_data.labels() };
            }
        }

        // call function -> should throw since we are out of resources
        EXPECT_THROW_WHAT_MATCHER((std::ignore = csvr.fit(training_data, plssvm::solver = solver)),
                                  plssvm::kernel_launch_resources,
                                  ::testing::ContainsRegex("Not enough device memory available on device.* even for the cg_implicit solver!"));
    }
}

TYPED_TEST(BaseCSVRFit, fit_device_memory_too_small) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // this test is only really applicable for the automatic solver type
    if constexpr (solver == plssvm::solver_type::automatic) {
        // create C-SVC: must be done using the mock class since the csvr base class is pure virtual
        const mock_csvr csvr{ plssvm::parameter{ plssvm::kernel_type = kernel } };

        // override on call
        using namespace plssvm::detail::literals;
        ON_CALL(csvr, get_device_memory()).WillByDefault(::testing::Return(std::vector<plssvm::detail::memory_size>{ 1_KiB, 1_KiB }));

        // clang-format off
        EXPECT_CALL(csvr, get_local_memory()).Times(1);
        EXPECT_CALL(csvr, get_device_memory()).Times(1);
        EXPECT_CALL(csvr, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvr, get_max_mem_alloc_size()).Times(0);
#endif
        EXPECT_CALL(csvr, assemble_kernel_matrix(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<const plssvm::parameter &>(),
                                ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                                ::testing::An<const std::vector<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>()))
                            .Times(0);
        EXPECT_CALL(csvr, blas_level_3(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                                ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>()))
                            .Times(0);
        // clang-format on

        // create data set
        plssvm::regression_data_set<label_type> training_data{ this->get_data_filename() };
        if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
            // chi-squared is well-defined for non-negative values only
            if (training_data.labels().has_value()) {
                training_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(training_data.data()), *training_data.labels() };
            }
        }

        // call function -> should throw since we are out of resources
        EXPECT_THROW_WHAT((std::ignore = csvr.fit(training_data, plssvm::solver = solver)),
                          plssvm::kernel_launch_resources,
                          "At least 512.00 MiB of memory must be available, but available are only 1.00 KiB!");
    }
}

TYPED_TEST(BaseCSVRFit, fit_local_memory_too_small) {
    using label_type = typename TestFixture::fixture_label_type;
    constexpr plssvm::solver_type solver = TestFixture::fixture_solver;
    constexpr plssvm::kernel_function_type kernel = TestFixture::fixture_kernel;

    // create C-SVC: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{ plssvm::parameter{ plssvm::kernel_type = kernel } };

    // override on call
    constexpr plssvm::detail::memory_size needed_local_mem_size = plssvm::detail::data_distribution::maximum_local_memory_needed();
    ON_CALL(csvr, get_local_memory()).WillByDefault(::testing::Return((std::vector<std::optional<plssvm::detail::memory_size>>{ std::make_optional(needed_local_mem_size / 2), std::make_optional(needed_local_mem_size / 2) })));

    EXPECT_CALL(csvr, get_local_memory()).Times(1);
    // this test is only really applicable for the automatic solver type
    if constexpr (solver == plssvm::solver_type::automatic) {
        // clang-format off
        EXPECT_CALL(csvr, get_device_memory()).Times(0);
        EXPECT_CALL(csvr, num_available_devices()).Times(0);
#if defined(PLSSVM_ENFORCE_MAX_MEM_ALLOC_SIZE)
        EXPECT_CALL(csvr, get_max_mem_alloc_size()).Times(0);
#endif
        EXPECT_CALL(csvr, assemble_kernel_matrix(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<const plssvm::parameter &>(),
                                ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                                ::testing::An<const std::vector<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>()))
                            .Times(0);
        EXPECT_CALL(csvr, blas_level_3(
                                ::testing::An<plssvm::solver_type>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<const std::vector<plssvm::detail::move_only_any> &>(),
                                ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                                ::testing::An<plssvm::real_type>(),
                                ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>()))
                            .Times(0);
        // clang-format on
    }

    // create data set
    plssvm::regression_data_set<label_type> training_data{ this->get_data_filename() };
    if constexpr (kernel == plssvm::kernel_function_type::chi_squared) {
        // chi-squared is well-defined for non-negative values only
        if (training_data.labels().has_value()) {
            training_data = plssvm::regression_data_set<label_type>{ util::matrix_abs(training_data.data()), *training_data.labels() };
        }
    }

    // call function -> should throw since we are out of resources
    EXPECT_THROW_WHAT((std::ignore = csvr.fit(training_data, plssvm::solver = solver)),
                      plssvm::kernel_launch_resources,
                      fmt::format("At least {} of local memory must be available, but available are only {}!",
                                  needed_local_mem_size,
                                  needed_local_mem_size / 2));
}

template <typename T>
class BaseCSVRPredict : public BaseCSVRMemberBase<T> { };

TYPED_TEST_SUITE(BaseCSVRPredict, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(BaseCSVRPredict, predict) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{};

    // for the C-SVR, every function will be called exactly once
    const int num_calls = 1;

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(1);
    EXPECT_CALL(csvr, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Return(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 1 })));
    // clang-format on

    // create data set and previously learned model
    const plssvm::regression_data_set<label_type> data_to_predict{ this->get_data_filename() };
    const plssvm::regression_model<label_type> learned_model{ this->get_model_filename() };

    // call function
    const std::vector<label_type> prediction = csvr.predict(learned_model, data_to_predict);
    EXPECT_EQ(prediction.size(), 6);
}

TYPED_TEST(BaseCSVRPredict, predict_num_feature_mismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set and previously learned model
    const plssvm::regression_data_set<label_type> data_to_predict{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };
    const plssvm::regression_model<label_type> learned_model{ this->get_model_filename() };

    // calling the function with mismatching number of features should throw
    EXPECT_THROW_WHAT(std::ignore = csvr.predict(learned_model, data_to_predict),
                      plssvm::invalid_parameter_exception,
                      "Number of features per data point (2) must match the number of features per support vector of the provided model (4)!");
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(BaseCSVRPredict, predict_communicator_mismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create mismatching MPI communicator
    MPI_Comm duplicated_mpi_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data set and previously learned model
    const plssvm::regression_data_set<label_type> data_to_predict{ this->get_data_filename() };
    const plssvm::regression_data_set<label_type> data_to_predict_wrong_comm{ comm, this->get_data_filename() };
    const plssvm::regression_model<label_type> learned_model{ this->get_model_filename() };
    const plssvm::regression_model<label_type> learned_model_wrong_comm{ comm, this->get_model_filename() };

    // calling the function with mismatching MPI communicators should throw
    EXPECT_THROW_WHAT(std::ignore = csvr.predict(learned_model_wrong_comm, data_to_predict),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVR and model must be identical!");
    EXPECT_THROW_WHAT(std::ignore = csvr.predict(learned_model, data_to_predict_wrong_comm),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVR and data set must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

template <typename T>
class BaseCSVRScore : public BaseCSVRMemberBase<T> { };

TYPED_TEST_SUITE(BaseCSVRScore, util::regression_label_type_gtest, naming::test_parameter_to_name);

TYPED_TEST(BaseCSVRScore, score_model) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{};

    // for the C-SVR, every function will be called exactly once
    const int num_calls = 1;

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(2);  // once for fit and once for score
    EXPECT_CALL(csvr, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>()))
                        .Times(num_calls)
                        .WillRepeatedly(::testing::Return(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 1 })));
    // clang-format on

    // create data set
    plssvm::regression_data_set<label_type> training_data{ this->get_data_filename() };

    // call function
    const plssvm::regression_model<label_type> model = csvr.fit(training_data,
                                                                plssvm::epsilon = 1e-10,
                                                                plssvm::max_iter = 20);

    // call function
    const plssvm::real_type score = csvr.score(model);
    EXPECT_LE(score, plssvm::real_type{ 1.0 });
}

TYPED_TEST(BaseCSVRScore, score_model_from_file) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{};

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, predict_values(
                    ::testing::An<const plssvm::parameter &>(),
                    ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                    ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                    ::testing::An<const std::vector<plssvm::real_type> &>(),
                    ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                    ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // read a previously learned model from a model file
    const plssvm::regression_model<label_type> learned_model{ this->get_model_filename() };

    // call function
    EXPECT_THROW_WHAT(std::ignore = csvr.score(learned_model), plssvm::invalid_parameter_exception, "The model must have labels to score it! Maybe to model was read from a LIBSVM model file?");
}

TYPED_TEST(BaseCSVRScore, score_data_set) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{};

    // for the C-SVR, every function will be called exactly once
    const int num_calls = 1;

    // mock the predict_values function
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(1);
    EXPECT_CALL(csvr, predict_values(
                        ::testing::An<const plssvm::parameter &>(),
                        ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                        ::testing::An<const std::vector<plssvm::real_type> &>(),
                        ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>()))
                    .Times(num_calls)
                    .WillRepeatedly(::testing::Return(util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ 6, 1 })));
    // clang-format on

    // create data set and previously learned model
    const plssvm::regression_data_set<label_type> data_to_score{ this->get_data_filename() };
    const plssvm::regression_model<label_type> learned_model{ this->get_model_filename() };

    // call function
    const plssvm::real_type score = csvr.score(learned_model, data_to_score);
    EXPECT_LE(score, plssvm::real_type{ 1.0 });
}

TYPED_TEST(BaseCSVRScore, score_data_set_no_label) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, predict_values(
                    ::testing::An<const plssvm::parameter &>(),
                    ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                    ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                    ::testing::An<const std::vector<plssvm::real_type> &>(),
                    ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                    ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set
    const plssvm::regression_data_set<label_type> data_to_score{ PLSSVM_TEST_PATH "/data/libsvm/3x2_without_label.libsvm" };
    // read a previously learned model from a model file
    const plssvm::regression_model<label_type> learned_model{ this->get_model_filename() };

    // in order to call score, the provided data set must contain labels
    EXPECT_THROW_WHAT(std::ignore = csvr.score(learned_model, data_to_score), plssvm::invalid_parameter_exception, "The data set to score must have labels!");
}

TYPED_TEST(BaseCSVRScore, score_data_set_num_features_mismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, predict_values(
                        ::testing::An<const plssvm::parameter &>(),
                        ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                        ::testing::An<const std::vector<plssvm::real_type> &>(),
                        ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                        ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set
    const std::vector<label_type> labels = util::get_correct_data_file_labels<label_type>();
    const auto data = util::generate_random_matrix<plssvm::aos_matrix<plssvm::real_type>>(plssvm::shape{ labels.size(), 2 });
    const plssvm::regression_data_set<label_type> data_to_score{ data, labels };

    // read a previously learned model from a model file
    const plssvm::regression_model<label_type> learned_model{ this->get_model_filename() };

    // calling the function with mismatching number of features should throw
    EXPECT_THROW_WHAT(std::ignore = csvr.score(learned_model, data_to_score),
                      plssvm::invalid_parameter_exception,
                      fmt::format("Number of features per data point ({}) must match the number of features per support vector of the provided model ({})!",
                                  data.num_cols(),
                                  learned_model.num_features()));
}

#if defined(PLSSVM_HAS_MPI_ENABLED)

TYPED_TEST(BaseCSVRScore, predict_communicator_mismatch) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvr base class is pure virtual
    const mock_csvr csvr{};

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(0);
    EXPECT_CALL(csvr, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create mismatching MPI communicator
    MPI_Comm duplicated_mpi_comm;
    MPI_Comm_dup(MPI_COMM_WORLD, &duplicated_mpi_comm);
    const plssvm::mpi::communicator comm{ duplicated_mpi_comm };

    // create data set and previously learned model
    const plssvm::regression_data_set<label_type> data_to_predict{ this->get_data_filename() };
    const plssvm::regression_data_set<label_type> data_to_predict_wrong_comm{ comm, this->get_data_filename() };
    const plssvm::regression_model<label_type> learned_model{ this->get_model_filename() };
    const plssvm::regression_model<label_type> learned_model_wrong_comm{ comm, this->get_model_filename() };

    // calling the function with mismatching MPI communicators should throw
    EXPECT_THROW_WHAT(std::ignore = csvr.score(learned_model_wrong_comm, data_to_predict),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVR and model must be identical!");
    EXPECT_THROW_WHAT(std::ignore = csvr.score(learned_model, data_to_predict_wrong_comm),
                      plssvm::mpi_exception,
                      "The MPI communicators provided to the C-SVR and data set must be identical!");

    MPI_Comm_free(&duplicated_mpi_comm);
}

#endif

TYPED_TEST(BaseCSVRScore, predict_local_memory_too_small) {
    using label_type = typename TestFixture::fixture_label_type;

    // create C-SVR: must be done using the mock class since the csvc base class is pure virtual
    const mock_csvr csvr{};

    // override on call
    constexpr plssvm::detail::memory_size needed_local_mem_size = plssvm::detail::data_distribution::maximum_local_memory_needed();
    ON_CALL(csvr, get_local_memory()).WillByDefault(::testing::Return((std::vector<std::optional<plssvm::detail::memory_size>>{ std::make_optional(needed_local_mem_size / 2), std::make_optional(needed_local_mem_size / 2) })));

    // mock the predict_values function -> since an exception should be triggered, the mocked function should never be called
    // clang-format off
    EXPECT_CALL(csvr, get_local_memory()).Times(1);
    EXPECT_CALL(csvr, predict_values(
                            ::testing::An<const plssvm::parameter &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const std::vector<plssvm::real_type> &>(),
                            ::testing::An<plssvm::aos_matrix<plssvm::real_type> &>(),
                            ::testing::An<const plssvm::aos_matrix<plssvm::real_type> &>())).Times(0);
    // clang-format on

    // create data set and previously learned model
    const plssvm::regression_data_set<label_type> data_to_predict{ this->get_data_filename() };
    const plssvm::regression_model<label_type> learned_model{ this->get_model_filename() };

    // calling the function with mismatching MPI communicators should throw
    EXPECT_THROW_WHAT(std::ignore = csvr.score(learned_model, data_to_predict),
                      plssvm::kernel_launch_resources,
                      fmt::format("At least {} of local memory must be available, but available are only {}!",
                                  needed_local_mem_size,
                                  needed_local_mem_size / 2));
}
