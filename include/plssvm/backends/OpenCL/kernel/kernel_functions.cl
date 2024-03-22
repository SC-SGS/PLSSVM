/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implement the different kernel functions on the GPU using OpenCL.
 */

#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// ATTENTION: if a new function is added here, the mapping in the 'create_command_queues' function must be updated accordingly!!!

//***************************************************//
//                 feature reductions                //
//***************************************************//

/**
 * @brief Compute the default feature reduction, i.e., a simple dot-product.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
real_type feature_reduce_dot(const real_type val1, const real_type val2) {
    return val1 * val2;
}

/**
 * @brief Compute the feature reduction for the radial basis function kernel function, i.e., the squared euclidean distance.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
real_type feature_reduce_euclidean_dist(const real_type val1, const real_type val2) {
    const real_type d = val1 - val2;
    return d * d;
}

/**
 * @brief Compute the feature reduction for the laplacian kernel function, i.e., the Manhattan distance.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
real_type feature_reduce_manhattan_dist(const real_type val1, const real_type val2) {
    return fabs(val1 - val2);
}

/**
 * @brief Compute the feature reduction for the chi-squared kernel function.
 * @note Be sure that the denominator isn't 0.0 which may be the case for padding values.
 * @param[in] val1 the first feature value
 * @param[in] val2 the second feature value
 * @return the reduced value (`[[nodiscard]]`)
 */
real_type feature_reduce_chi_squared(const real_type val1, const real_type val2) {
    const real_type s = val1 + val2;
    if (s == 0.0) {
        return 0.0;
    } else {
        const real_type d = val1 - val2;
        return (d * d) / s;
    }
}

//***************************************************//
//                  kernel functions                 //
//***************************************************//

/**
 * @brief Compute the linear kernel function using @p value.
 * @param[in] value the value to apply the linear kernel function to
 * @return the result value (`[[nodiscard]]`)
 */
real_type apply_linear_kernel_function(const real_type value) {
    return value;
}

/**
 * @brief Compute the polynomial kernel function using @p value.
 * @param[in] value the value to apply the polynomial kernel function to
 * @param[in] degree the degree parameter of the polynomial kernel function
 * @param[in] gamma the gamma parameter of the polynomial kernel function
 * @param[in] coef0 the coef0 parameter of the polynomial kernel function
 * @return the result value (`[[nodiscard]]`)
 */
real_type apply_polynomial_kernel_function(const real_type value, const int degree, const real_type gamma, const real_type coef0) {
    return pow(gamma * value + coef0, degree);
}

/**
 * @brief Compute the radial basis function kernel function using @p value.
 * @param[in] value the value to apply the rbf kernel function to
 * @param[in] gamma the gamma parameter of the rbf kernel function
 * @return the result value (`[[nodiscard]]`)
 */
real_type apply_rbf_kernel_function(const real_type value, const real_type gamma) {
    return exp(-gamma * value);
}

/**
 * @brief Compute the sigmoid kernel function using @p value.
 * @param[in] value the value to apply the sigmoid kernel function to
 * @param[in] gamma the gamma parameter of the kernel kernel function
 * @param[in] coef0 the coef0 parameter of the kernel kernel function
 * @return the result value (`[[nodiscard]]`)
 */
real_type apply_sigmoid_kernel_function(const real_type value, const real_type gamma, const real_type coef0) {
    return tanh(gamma * value + coef0);
}

/**
 * @brief Compute the laplacian function kernel function using @p value.
 * @param[in] value the value to apply the laplacian kernel function to
 * @param[in] gamma the gamma parameter of the laplacian kernel function
 * @return the result value (`[[nodiscard]]`)
 */
real_type apply_laplacian_kernel_function(const real_type value, const real_type gamma) {
    return exp(-gamma * value);
}

/**
 * @brief Compute the chi-squared function kernel function using @p value.
 * @param[in] value the value to apply the chi-squared kernel function to
 * @param[in] gamma the gamma parameter of the chi-squared kernel function
 * @return the result value (`[[nodiscard]]`)
 */
real_type apply_chi_squared_kernel_function(const real_type value, const real_type gamma) {
    return exp(-gamma * value);
}
