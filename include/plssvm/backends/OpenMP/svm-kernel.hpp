/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines the kernel functions for the C-SVM using the OpenMP backend.
 */

#pragma once

#include "plssvm/exceptions/exceptions.hpp"  // plssvm::not_implemented_exception

#include <vector>  // std::vector

namespace plssvm::openmp {

/**
 * @brief Calculates the C-SVM kernel using the linear kernel function.
 * @tparam real_type the type of the data
 * @param[in] q the `q` vector
 * @param[in] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] data the data matrix
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 */
template <typename real_type>
void device_kernel_linear(const std::vector<real_type> &q, std::vector<real_type> &ret, const std::vector<real_type> &d, const std::vector<std::vector<real_type>> &data, real_type QA_cost, real_type cost, int add);

/**
 * @brief Calculates the C-SVM kernel using the polynomial kernel function.
 * @tparam real_type the type of the data
 * @param[in] data the data matrix
 * @param[in] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] degree the degree parameter used in the polynomial kernel function
 * @param[in] gamma the gamma parameter used in the polynomial kernel function
 * @param[in] coef0 the coef0 parameter used in the polynomial kernel function
 *
 * @attention Currently not implemented!
 */
template <typename real_type>
void device_kernel_poly(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &ret, const std::vector<real_type> &d, real_type QA_cost, real_type cost, int add, real_type gamma, real_type coef0, real_type degree) {
    throw not_implemented_exception{ "The polynomial kernel is currently not implemented for the OpenMP backend!" };
}

/**
 * @brief Calculates the C-SVM kernel using the radial basis function kernel function.
 * @tparam real_type the type of the data
 * @param[in] data the data matrix
 * @param[in] ret the result vector
 * @param[in] d the right-hand side of the equation
 * @param[in] QA_cost he bottom right matrix entry multiplied by cost
 * @param[in] cost 1 / the cost parameter in the C-SVM
 * @param[in] add denotes whether the values are added or subtracted from the result vector
 * @param[in] gamma the gamma parameter used in the rbf kernel function
 *
 * @attention Currently not implemented!
 */
template <typename real_type>
void device_kernel_radial(const std::vector<std::vector<real_type>> &data, std::vector<real_type> &ret, const std::vector<real_type> &d, real_type QA_cost, real_type cost, int add, real_type gamma) {
    throw not_implemented_exception{ "The radial basis function kernel is currently not implemented for the OpenMP backend!" };
}

//template <typename T>
//void kernel_linear(const std::vector<T> &b, const std::vector<std::vector<T>> &data, const T *datlast, const T *q_d, std::vector<T> &Ad, const T *d, std::size_t dim, T QA_cost, T cost, int sgn);

/*
void kernel_linear(std::tuple<int,int>, std::tuple<int,int>, real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add);
void kernel_poly(std::tuple<int,int>, std::tuple<int,int>, real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma, const real_t coef0 ,const real_t degree);
void kernel_radial(std::tuple<int,int>, std::tuple<int,int>, real_t *q, real_t *ret, real_t *d, real_t *data_d,const real_t QA_cost, const real_t cost,const int Ncols,const int Nrows,const int add, const real_t gamma);
*/

}  // namespace plssvm::openmp