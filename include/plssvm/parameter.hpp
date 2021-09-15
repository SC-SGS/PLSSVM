/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements the base class encapsulating all necessary parameters.
 */

#pragma once

#include "plssvm/backend_types.hpp"    // plssvm::backend_type
#include "plssvm/kernel_types.hpp"     // plssvm::kernel_type
#include "plssvm/target_platform.hpp"  // plssvm::target_platform

#include "fmt/ostream.h"  // use operator<< to enable fmt::format with custom type

#include <memory>       // std::shared_ptr
#include <string>       // std::string
#include <type_traits>  // std::is_same_v

namespace plssvm {

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
    /// Unsigned integer type.
    using size_type = std::size_t;

    /// virtual, default destructor.
    virtual ~parameter() = default;

    /**
     * @brief Parse a file in the [libsvm sparse file format](https://www.csie.ntu.edu.tw/~cjlin/libsvm/faq.html#f303).
     * @details The sparse libsvm file format saves each data point with its respective class as follows:
     * @code
     * <label> <index1>:<value1> <index2>:<value2> ... <indexN>:<valueN>
     * @endcode
     * Only non-empty lines that don't start with `#` (= optional comments) are parsed.
     *
     * An example libsvm file could look as follows:
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
     * @param[in] filename name of the libsvm file to parse
     */
    void parse_libsvm(const std::string &filename);
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
     */
    void parse_arff(const std::string &filename);
    /**
     * Parse a model file in the LIBSVM model file format.
     * @param filename the model file to parse
     */
    void parse_model_file(const std::string &filename);
    /**
     * @brief Parse the given file. If the file is in the arff format (has the `.arff` extension), the arff parser is used, otherwise the libsvm parser is used.
     * @param[in] filename name of the file to parse
     */
    void parse_file(const std::string &filename);

    /// The used kernel function: linear, polynomial or radial basis functions (rbf).
    kernel_type kernel = kernel_type::linear;
    /// The degree parameter used in the polynomial kernel function.
    int degree = 3;
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    real_type gamma = 0.0;
    /// The coef0 parameter used in the polynomial kernel function.
    real_type coef0 = 0.0;
    /// The cost parameter in the C-SVM.
    real_type cost = 1.0;
    /// The error tolerance parameter for the CG algorithm.
    real_type epsilon = 0.001;
    /// If `true` additional information (e.g. timing information) will be printed during execution.
    bool print_info = true;
    /// The used backend: OpenMP, OpenCL, CUDA, or SYCL.
    backend_type backend = backend_type::openmp;
    /// The target platform: automatic (depending on the used backend), CPUs or GPUs from NVIDIA, AMD or Intel.
    target_platform target = target_platform::automatic;

    /// The name of the data file to parse.
    std::string input_filename{};
    /// The name of the model file to write the learned Support Vectors to.
    std::string model_filename{};
    /// The name of the file to write the prediction to.
    std::string predict_filename{};

    /// The data used the train the SVM.
    std::shared_ptr<const std::vector<std::vector<real_type>>> data_ptr{};
    /// The labels associated to each data point.
    std::shared_ptr<const std::vector<real_type>> value_ptr{};
    /// The weights associated to each data point after training.
    std::shared_ptr<const std::vector<real_type>> alphas_ptr{};
    /// The data to predict.
    std::shared_ptr<const std::vector<std::vector<real_type>>> test_data_ptr{};

    real_type rho = 0.0;

  protected:
    /*
     * Generate model filename based on the name of the input file.
     */
    std::string model_name_from_input();
};

extern template class parameter<float>;
extern template class parameter<double>;

/**
 * @brief Stream-insertion operator overload for convenient printing of all parameters encapsulated by @p params.
 * @tparam T the type of the data
 * @param[in,out] out the output-stream to write the kernel type to
 * @param[in] params the parameters
 * @return the output-stream
 */
template <typename T>
std::ostream &operator<<(std::ostream &out, const parameter<T> &params);

}  // namespace plssvm