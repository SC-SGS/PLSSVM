/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements the parameter base class encapsulating all necessary parameters.
 */

#pragma once

#include "plssvm/backend_types.hpp"                         // plssvm::backend_type
#include "plssvm/backends/SYCL/implementation_type.hpp"     // plssvm::sycl_generic::implementation_type
#include "plssvm/backends/SYCL/kernel_invocation_type.hpp"  // plssvm::sycl_generic::kernel_invocation_type
#include "plssvm/kernel_types.hpp"                          // plssvm::kernel_type
#include "plssvm/target_platforms.hpp"                      // plssvm::target_platform

#include <iosfwd>       // forward declare std::ostream
#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

namespace plssvm {

namespace sycl {
using namespace ::plssvm::sycl_generic;
}

/**
 * @brief Base class for encapsulating all necessary parameters possibly provided through command line arguments.
 * @tparam T the type of the data
 */
template <typename T>
class parameter {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;

    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~parameter() = default;

    /**
     * @brief Parse a file in the [LIBSVM sparse file format](https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f303).
     * @details The sparse LIBSVM file format saves each data point with its respective class as follows:
     * @code
     * <label> <index1>:<value1> <index2>:<value2> ... <indexN>:<valueN>
     * @endcode
     * Only non-empty lines that don't start with `#` (= optional comments) are parsed.
     *
     * An example LIBSVM file could look as follows:
     * @code
     * # this is a comment
     *  1 1:1.29801019287324655 2:0.51687296029754564
     * -1 1:1.01405596624706053
     * -1 1:0.60276937379453293 3:-0.13086851759108944
     * -1 2:0.298499933047586044 # this is also a comment
     * @endcode
     *
     * Be aware that the parsed output is **always** in a dense format. The above file for example will be parsed to a
     * @code
     * std::vector<std::vector<real_type>> data = {
     *   { 1.29801019287324655, 0.51687296029754564, 0.0 },
     *   { 1.01405596624706053, 0.0, 0.0 },
     *   { 0.60276937379453293, 0.0, -0.13086851759108944 },
     *   { 0.0, 0.298499933047586044, 0.0 }
     * }
     * @endcode
     *
     * If possible, uses a memory mapped file internally to speed up the file parsing.
     * @param[in] filename name of the LIBSVM file to parse
     * @param[in] data_ptr_ref the underlying matrix to save the parsed values to
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     * @throws plssvm::invalid_file_format_exception if the @p filename has an invalid format (e.g. an empty file, a file not using the LIBSVM file format, ...)
     */
    void parse_libsvm_file(const std::string &filename, std::shared_ptr<const std::vector<std::vector<real_type>>> &data_ptr_ref);
    /**
     * @brief Parse a file in the [arff file format](https://www.cs.waikato.ac.nz/ml/weka/arff.html).
     * @details The arff file format saves each data point with its respective class as follows:
     * @code
     * <value1>,<value2>,...,<valueN>,<label>
     * @endcode
     * Additionally, the sparse arff file format (or a mix of both) is also supported:
     * @code
     * {<index1> <value1>, <index2> <value2>, <label>}
     * @endcode
     * Only non-empty lines that don't start with `%` (= optional comments) are parsed.
     *
     * An example arff file could look as follows:
     * @code
     * % Title
     * % comments
     * @RELATION name
     * @ATTRIBUTE first    NUMERIC
     * @ATTRIBUTE second   numeric
     * @ATTRIBUTE third    Numeric
     * @ATTRIBUTE fourth   NUMERIC
     * @ATTRIBUTE class    NUMERIC
     *
     * @DATA
     * -1.117827500607882,-2.9087188881250993,0.66638344270039144,1.0978832703949288,1
     * -0.5282118298909262,-0.335880984968183973,0.51687296029754564,0.54604461446026,1
     * {1 0.60276937379453293, 2 -0.13086851759108944, 4 -1}
     * {0 1.88494043717792, 1 1.00518564317278263, 2 0.298499933047586044, 3 1.6464627048813514, 4 -1}
     * @endcode
     * The necessary arff header values must be present and the type of the `@ATTRIBUTE` tags must be `NUMERIC`.
     *
     * Be aware that the parsed output is **always** in a dense format. The above file for example will be parsed to a
     * @code
     * std::vector<std::vector<real_type>> data = {
     *   { -1.117827500607882, -2.9087188881250993, 0.66638344270039144, 1.0978832703949288 },
     *   { -0.5282118298909262, -0.335880984968183973, 0.51687296029754564, 0.54604461446026 },
     *   { 0.0, 0.60276937379453293, -0.13086851759108944, 0.0 },
     *   { 1.88494043717792, 1.00518564317278263, 0.298499933047586044, 1.6464627048813514 }
     * }
     * @endcode
     *
     * If possible, uses a memory mapped file internally to speed up the file parsing.
     * @param[in] filename name of the arff file to parse
     * @param[in] data_ptr_ref the underlying matrix to save the parsed values to
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     * @throws plssvm::invalid_file_format_exception if the @p filename has an invalid format (e.g. an empty file, invalid arff header, ...)
     */
    void parse_arff_file(const std::string &filename, std::shared_ptr<const std::vector<std::vector<real_type>>> &data_ptr_ref);
    /**
     * @brief Parse a model file in the [LIBSVM model file format](https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f402).
     * @details An example LIBSVM file could look as follows:
     * @code
     * svm_type c_svc
     * kernel_type linear
     * nr_class 2
     * total_sv 5
     * rho 0.37330625882191915
     * label 1 -1
     * nr_sv 2 3
     * SV
     * -0.17609610490769723 0:-1.117828e+00 1:-2.908719e+00 2:6.663834e-01 3:1.097883e+00
     * 0.8838187731213127 0:-5.282118e-01 1:-3.358810e-01 2:5.168730e-01 3:5.460446e-01
     * -0.47971257671001616 0:-2.098121e-01 1:6.027694e-01 2:-1.308685e-01 3:1.080525e-01
     * 0.0034556484621847128 0:1.884940e+00 1:1.005186e+00 2:2.984999e-01 3:1.646463e+00
     * -0.23146573996578407 0:5.765022e-01 1:1.014056e+00 2:1.300943e-01 3:7.261914e-01
     * @endcode
     * @param[in] filename the model file to parse
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     * @throws plssvm::invalid_file_format_exception if the @p filename has an invalid format (e.g. an empty file, invalid LIBSVM model file header, ...)
     */
    void parse_model_file(const std::string &filename);
    /**
     * @brief Parse the given file. If the file is in the arff format (has the `.arff` extension), the arff parser is used, otherwise the LIBSVM parser is used.
     * @param[in] filename name of the file to parse
     * @param[in] data_ptr_ref the underlying matrix to save the parsed values to
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     * @throws plssvm::invalid_file_format_exception if the @p filename has an invalid format (e.g. an empty file, ...)
     */
    void parse_file(const std::string &filename, std::shared_ptr<const std::vector<std::vector<real_type>>> &data_ptr_ref);

    /**
     * @brief Parse the given file as training data. If the file is in the arff format (has the `.arff` extension), the arff parser is used, otherwise the LIBSVM parser is used.
     * @details Saves the data to the member variable #data_ptr.
     * @param[in] filename name of the file to parse
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     * @throws plssvm::invalid_file_format_exception if the @p filename has an invalid format (e.g. an empty file, ...)
     */
    void parse_train_file(const std::string &filename);
    /**
     * @brief Parse the given file as test data. If the file is in the arff format (has the `.arff` extension), the arff parser is used, otherwise the LIBSVM parser is used.
     * @details Saves the data to the member variable #test_data_ptr.
     * @param[in] filename name of the file to parse
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     * @throws plssvm::invalid_file_format_exception if the @p filename has an invalid format (e.g. an empty file, ...)
     */
    void parse_test_file(const std::string &filename);

    /// The used kernel function: linear, polynomial or radial basis functions (rbf).
    kernel_type kernel = kernel_type::linear;
    /// The degree parameter used in the polynomial kernel function.
    int degree = 3;
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    real_type gamma = real_type{ 0.0 };
    /// The coef0 parameter used in the polynomial kernel function.
    real_type coef0 = real_type{ 0.0 };
    /// The cost parameter in the C-SVM.
    real_type cost = real_type{ 1.0 };
    /// The error tolerance parameter for the CG algorithm.
    real_type epsilon = static_cast<real_type>(0.001);
    /// If `true` additional information (e.g. timings) will be printed during execution.
    bool print_info = true;
    /// The used backend: automatic (depending on the specified target_platforms), OpenMP, OpenCL, CUDA, or SYCL.
    backend_type backend = backend_type::automatic;
    /// The target platform: automatic (depending on the used backend), CPUs or GPUs from NVIDIA, AMD or Intel.
    target_platform target = target_platform::automatic;

    /// The kernel invocation type when using SYCL as backend.
    sycl::kernel_invocation_type sycl_kernel_invocation_type = sycl::kernel_invocation_type::automatic;
    /// The SYCL implementation to use with --backend=sycl.
    sycl::implementation_type sycl_implementation_type = sycl::implementation_type::automatic;

    /// The name of the data/test file to parse.
    std::string input_filename{};
    /// The name of the model file to write the learned support vectors to/to parse the saved model from.
    std::string model_filename{};
    /// The name of the file to write the prediction to.
    std::string predict_filename{};

    /// The data used the train the SVM.
    std::shared_ptr<const std::vector<std::vector<real_type>>> data_ptr{};
    /// The labels associated with each data point.
    std::shared_ptr<const std::vector<real_type>> value_ptr{};
    /// The weights associated with each data point after training.
    std::shared_ptr<const std::vector<real_type>> alpha_ptr{};
    /// The test data to predict.
    std::shared_ptr<const std::vector<std::vector<real_type>>> test_data_ptr{};

    /// The rho value of the calculated/read model.
    real_type rho = real_type{ 0.0 };

  protected:
    /**
     * @brief Generate a model filename based on the name of the input file.
     * @return `${input_filename}.model` (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string model_name_from_input();
    /**
     * @brief Generate a predict filename based on the name of the input file.
     * @return `${input_filename}.predict` (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string predict_name_from_input();
};

extern template class parameter<float>;
extern template class parameter<double>;

/**
 * @brief Output all parameters encapsulated by @p params to the given output-stream @p out.
 * @tparam T the type of the data
 * @param[in,out] out the output-stream to write the parameters to
 * @param[in] params the parameters
 * @return the output-stream
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params);

}  // namespace plssvm