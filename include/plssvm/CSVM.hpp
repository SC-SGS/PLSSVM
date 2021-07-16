/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Defines the base class for all C-SVM backends and implements the functionality shared by all of them.
 */

#pragma once

#include "plssvm/kernel_types.hpp"  // plssvm::kernel_type
#include "plssvm/parameter.hpp"     // plssvm::parameter

#include <cstddef>      // std::size_t
#include <string>       // std::string
#include <type_traits>  // std::is_same_v
#include <vector>       // std::vector

namespace plssvm {

/**
 * @brief Base class for all C-SVM backends.
 * @tparam T the type of the data
 */
template <typename T>
class CSVM {
    // only float and doubles are allowed
    static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>, "The template type can only be 'float' or 'double'!");

  public:
    /// The type of the data. Must be either `float` or `double`.
    using real_type = T;
    /// Unsigned integer type.
    using size_type = std::size_t;

    //*************************************************************************************************************************************//
    //                                                      special member functions                                                       //
    //*************************************************************************************************************************************//
    /**
     * @brief Construct a new C-SVM with the parameters given through @p params.
     * @param[in] params struct encapsulating all possible parameters
     */
    explicit CSVM(const parameter<T> &params);
    /**
     * @brief Construct an new C-SVM explicitly specifying all necessary parameters.
     * @param[in] kernel the type of the kernel function
     * @param[in] degree parameter used in the polynomial kernel function
     * @param[in] gamma parameter used in the polynomial and rbf kernel functions
     * @param[in] coef0 parameter use din the polynomial kernel function
     * @param[in] cost parameter of the C-SVM
     * @param[in] epsilon error tolerance in the CG algorithm
     * @param[in] print_info if `true` additional information will be printed during execution
     */
    CSVM(kernel_type kernel, real_type degree, real_type gamma, real_type coef0, real_type cost, real_type epsilon, bool print_info);

    /**
     * @brief Virtual destructor to enable safe inheritance.
     */
    virtual ~CSVM() = default;

    /**
     * @brief Disable copy-constructor.
     */
    CSVM(const CSVM &) = delete;
    // clang-format off
    /**
     * @brief Explicitly allow move-construction.
     */
    CSVM(CSVM &&) noexcept = default;
    // clang-format on

    //*************************************************************************************************************************************//
    //                                                             IO functions                                                            //
    //*************************************************************************************************************************************//
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
     * @brief Parse the given file. If the file is in the arff format (has the `.arff` extension), the arff parser is used, otherwise the libsvm parser is used.
     * @param[in] filename name of the file to parse
     */
    void parse_file(const std::string &filename);
    /**
     * @brief Write the calculated model to the given file.
     * @details Writes the model using the libsvm format:
     * @code
     * svm_type c_svc
     * kernel_type linear
     * nr_class 2
     * total_sv 5
     * rho 0.37332362
     * label 1 -1
     * nr_sv 2 3
     * SV
     * -0.17609704 0:-1.117828e+00 1:-2.908719e+00 2:6.663834e-01 3:1.097883e+00
     * 0.883819 0:-5.282118e-01 1:-3.358810e-01 2:5.168729e-01 3:5.460446e-01
     * -0.47971326 0:-2.098121e-01 1:6.027694e-01 2:-1.308685e-01 3:1.080525e-01
     * -0.23146635 0:5.765022e-01 1:1.014056e+00 2:1.300943e-01 3:7.261914e-01
     * 0.0034576654 0:1.884940e+00 1:1.005186e+00 2:2.984999e-01 3:1.646463e+00
     * @endcode
     * @param[in] filename name of the file to write the model information to
     */
    void write_model(const std::string &filename);

    //*************************************************************************************************************************************//
    //                                                             learn model                                                             //
    //*************************************************************************************************************************************//
    /**
     * @brief Learns the Support Vectors given the data in @p input_filename and writes the results to @p model_filename.
     * @details Performs 4 steps:
     * 1. Read and parse the data file
     * 2. Load the data onto the used device (e.g. one or more GPUs)
     * 3. Learn the model by solving a minimization problem using the Conjugated Gradients algorithm
     * 4. Write the results to the model file
     * @param[in] input_filename name of the data file to parse
     * @param[in] model_filename name of the model file to write the model information to
     */
    void learn(const std::string &input_filename, const std::string &model_filename);

  protected:
    /**
     * @brief Learns the Support Vectors previously parsed.
     * @details Learn the model by solving a minimization problem using the Conjugated Gradients algorithm.
     */
    void learn();  // TODO: public after correct exception handling

  public:
    //*************************************************************************************************************************************//
    //                                                               predict                                                               //
    //*************************************************************************************************************************************//
    // TODO: protected?
    //    virtual std::vector<real_type> predict(real_type *, size_type, size_type) = 0;

    //*************************************************************************************************************************************//
    //                                                               getter                                                                //
    //*************************************************************************************************************************************//
    // TODO: other getter?
    //    [[nodiscard]] real_type get_bias() const noexcept { return bias_; };

  protected:
    //*************************************************************************************************************************************//
    //                                         pure virtual, must be implemented by all subclasses                                         //
    //*************************************************************************************************************************************//
    /**
     * @brief Initialize the data on the respective device(s) (e.g. GPUs).
     */
    virtual void setup_data_on_device() = 0;
    /**
     * @brief Generate the vector `q`, a subvector of the least-squares matrix equation.
     * @return the generated `q` vector
     */
    [[nodiscard]] virtual std::vector<real_type> generate_q() = 0;
    /**
     * @brief Solves the equation \f$Ax = b\f$ using the Conjugated Gradients algorithm.
     * @details Solves using a slightly modified version of the CG algorithm described by [Jonathan Richard Shewchuk](https://www.cs.cmu.edu/~quake-papers/painless-conjugate-gradient.pdf):
     * \image html cg.png
     * @param[in] b the right-hand side of the equation \f$Ax = b\f$
     * @param[in] imax the maximum number of CG iterations
     * @param[in] eps error tolerance
     * @param[in] q subvector of the least-squares matrix equation
     * @return `x`
     */
    virtual std::vector<real_type> solver_CG(const std::vector<real_type> &b, size_type imax, real_type eps, const std::vector<real_type> &q) = 0;
    /**
     * @brief TODO:
     */
    virtual void load_w() = 0;  // TODO: implemented together with predict

    //*************************************************************************************************************************************//
    //                                                          kernel functions                                                           //
    //*************************************************************************************************************************************//
    /**
     * @brief Computes the value of the two arrays denoted by the pointers @p xi and @p xj using the kernel function specified during construction.
     * @param[in] xi the first array
     * @param[in] xj the seconds array
     * @param[in] dim the length of both arrays
     * @throws unsupported_kernel_type_exception if the kernel_type cannot be recognized
     * @return the value computed by the kernel function
     */
    real_type kernel_function(const real_type *xi, const real_type *xj, size_type dim);
    /**
     * @brief Computes the value of the two vectors @p xi and @p xj using the kernel function specified during construction.
     * @param[in] xi the first vector
     * @param[in] xj the second vector
     * @throws unsupported_kernel_type_exception if the kernel_type cannot be recognized
     * @return the value computed by the kernel function
     */
    real_type kernel_function(const std::vector<real_type> &xi, const std::vector<real_type> &xj);

    /**
     * @brief Transforms the 2D data from AoS to a 1D SoA layout, ignoring the last data point and adding boundary points.
     * @param[in] boundary the number of boundary cells
     * @attention boundary values can contain random numbers
     * @return an 1D vector in a SoA layout
     */
    std::vector<real_type> transform_data(size_type boundary);

    //*************************************************************************************************************************************//
    //                                              parameter initialized by the constructor                                               //
    //*************************************************************************************************************************************//
    /// The used kernel function: linear, polynomial or radial basis functions (rbf).
    const kernel_type kernel_;
    /// The degree parameter used in the polynomial kernel function.
    const real_type degree_;
    /// The gamma parameter used in the polynomial and rbf kernel functions.
    real_type gamma_;
    /// The coef0 parameter used in the polynomial kernel function.
    const real_type coef0_;
    /// The cost parameter in the C-SVM.
    real_type cost_;
    /// The error tolerance parameter for the CG algorithm.
    const real_type epsilon_;
    /// If `true` additional information (e.g. timing information) will be printed during execution.
    const bool print_info_;

    //*************************************************************************************************************************************//
    //                                                         internal variables                                                          //
    //*************************************************************************************************************************************//
    /// The number of data points in the data set.
    size_type num_data_points_{};
    /// The number of features per data point.
    size_type num_features_{};
    /// The data used the train the SVM.
    std::vector<std::vector<real_type>> data_{};
    /// The labels associated to each data point.
    std::vector<real_type> value_{};
    /// The bias after learning.
    real_type bias_{};
    /// The bottom right matrix entry multiplied by cost.
    real_type QA_cost_{};
    /// The result of the CG calculation.
    std::vector<real_type> alpha_{};
};

extern template class CSVM<float>;
extern template class CSVM<double>;

}  // namespace plssvm