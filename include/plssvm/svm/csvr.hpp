/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Defines the base class for all C-SVM backends and implements the functionality shared by all of them for the regression task.
 */

#ifndef PLSSVM_SVM_CSVR_HPP_
#define PLSSVM_SVM_CSVR_HPP_
#pragma once

#include "plssvm/data_set/regression_data_set.hpp"  // plssvm::regression_data_set
#include "plssvm/svm/csvm.hpp"                      // plssvm::csvm

namespace plssvm {

class csvr : virtual public csvm {
  public:
    //*************************************************************************************************************************************//
    //                                                              fit model                                                              //
    //*************************************************************************************************************************************//
    /**
     * @brief Fit a model using the current SVM on the @p data using the provided multi-class classification strategy.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @tparam Args the type of the potential additional parameters
     * @param[in] data the data used to train the SVM model
     * @param[in] named_args the potential additional parameters (`epsilon`, `max_iter`, and `classification`)
     * @throws plssvm::invalid_parameter_exception if the provided value for `epsilon` is greater or equal than zero
     * @throws plssvm::invlaid_parameter_exception if the provided maximum number of iterations is less or equal than zero
     * @throws plssvm::invalid_parameter_exception if the training @p data does **not** include labels
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::solve_lssvm_system_of_linear_equations`
     * @note For binary classification **always** one vs. all is used regardless of the provided parameter!
     * @return the learned model (`[[nodiscard]]`)
     */
    template <typename label_type, typename... Args>
    [[nodiscard]] model<label_type> fit(const regression_data_set<label_type> &data, Args &&...named_args) const {
        std::cerr << "REGRESSION FIT" << std::endl;
        return csvm::fit(data, std::forward<Args>(named_args)...);
    }

    //*************************************************************************************************************************************//
    //                                                          predict and score                                                          //
    //*************************************************************************************************************************************//
    /**
     * @brief Predict the labels for the @p data set using the @p model.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learned model
     * @param[in] data the data to predict the labels for
     * @throws plssvm::invalid_parameter_exception if the number of features in the @p model's support vectors don't match the number of features in the @p data set
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @return the predicted labels (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] std::vector<label_type> predict(const model<label_type> &model, const regression_data_set<label_type> &data) const {
        std::cerr << "REGRESSION PREDICT" << std::endl;
        return csvm::predict(model, data);
    }

    /**
     * @brief Calculate the accuracy of the labeled @p data set using the @p model.
     * @details Uses the one vs. all (OAA) for the multi-class classification task.
     * @tparam label_type the type of the label (an arithmetic type or `std::string`)
     * @param[in] model a previously learned model
     * @param[in] data the labeled data set to score
     * @throws plssvm::invalid_parameter_exception if the @p data to score has no labels
     * @throws plssvm::invalid_parameter_exception if the number of features in the @p model's support vectors don't match the number of features in the @p data set
     * @throws plssvm::exception any exception thrown in the respective backend's implementation of `plssvm::csvm::predict_values`
     * @return the accuracy of the labeled @p data (`[[nodiscard]]`)
     */
    template <typename label_type>
    [[nodiscard]] real_type score(const model<label_type> &model, const regression_data_set<label_type> &data) const {
        std::cerr << "REGRESSION SCORE" << std::endl;
        return csvm::score(model, data);
    }
};

}  // namespace plssvm

#endif  // PLSSVM_SVM_CSVR_HPP_
