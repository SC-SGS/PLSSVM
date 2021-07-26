/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines a C-SVM using the SYCL backend.
 */

#pragma once

#include "plssvm/csvm.hpp"          // plssvm::csvm
#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type
#include "plssvm/parameter.hpp"     // plssvm::parameter

#include <vector>  // std::vector

namespace plssvm::sycl {

/**
 * @brief The C-SVM class using the SYCL backend.
 * @tparam T the type of the data
 */
template <typename T>
class csvm : ::plssvm::csvm<T> {
  protected:
    // protected for the test MOCK class
    /// The template base type of the CUDA_SVM class.
    using base_type = ::plssvm::csvm<T>;

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = typename base_type::real_type;
    /// Unsigned integer type.
    using size_type = typename base_type::size_type;

    /**
     * @brief Construct a new C-SVM using the SYCL backend with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit csvm(const parameter<T> &params);
    /**
     * @brief Construct an new C-SVM using the SYCL backend explicitly specifying all necessary parameters.
     * @param[in] kernel the type of the kernel function
     * @param[in] degree parameter used in the polynomial kernel function
     * @param[in] gamma parameter used in the polynomial and rbf kernel functions
     * @param[in] coef0 parameter use din the polynomial kernel function
     * @param[in] cost parameter of the C-SVM
     * @param[in] epsilon error tolerance in the CG algorithm
     * @param[in] print_info if `true` additional information will be printed during execution
     */
    csvm(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

    // std::vector<real_type> predict(real_type *, size_type, size_type) override;  // TODO: implement

  protected:
    void setup_data_on_device() override;
    std::vector<real_type> generate_q() override;
    std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) override;
    void load_w() override {}  // TODO: implement
};

extern template class csvm<float>;
extern template class csvm<double>;

}  // namespace plssvm::sycl