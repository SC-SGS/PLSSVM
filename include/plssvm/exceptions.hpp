/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements custom exception classes derived from `std::exception` including source location information.
 */

#pragma once

#include <fmt/format.h>  // fmt::format

#include <ostream>      // std::ostream
#include <stdexcept>    // std::runtime_error
#include <string_view>  // std::string_view

namespace plssvm {

/**
 * @brief The `plssvm::source_location` class represents certain information about the source code, such as file names, line numbers or function names.
 * @details Based on [`std::source_location`](https://en.cppreference.com/w/cpp/utility/source_location).
 */
class source_location {
  public:
    /**
     * @brief Construct new source location information about the current call side.
     * @param[in] file_name the file name including its absolute path, as given by `__builtin_FILE()`
     * @param[in] function_name the function name (without return type and parameters), as given by `__builtin_FUNCTION()`
     * @param[in] line the line number, as given by `__builtin_LINE()`
     * @param[in] column the column number, always `0`
     * @return the source location object holding the information about the current call side (`[[nodiscard]]`)
     */
    [[nodiscard]] static source_location current(
        const char *file_name = __builtin_FILE(),
        const char *function_name = __builtin_FUNCTION(),
        const int line = __builtin_LINE(),
        const int column = 0) noexcept {
        source_location loc;

        loc.file_name_ = file_name;
        loc.function_name_ = function_name;
        loc.line_ = line;
        loc.column_ = column;

        return loc;
    }

    /**
     * @brief Returns the absolute path name of the file or `"unknown"` if no information could be retrieved.
     * @return the file name (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string_view function_name() const noexcept { return function_name_; }
    /**
     * @brief Returns the function name without additional signature information (i.e. return type and parameters)
     *        or `"unknown"` if no information could be retrieved.
     * @return the function name (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string_view file_name() const noexcept { return file_name_; }
    /**
     * @brief Returns the line number or `0` if no information could be retrieved.
     * @return the line number (`[[nodiscard]]`)
     * @return
     */
    [[nodiscard]] int line() const noexcept { return line_; }
    /**
     * @brief Returns the column number. Always `0`!
     * @return `0` (`[[nodiscard]]`)
     */
    [[nodiscard]] int column() const noexcept { return column_; }

  private:
    std::string_view function_name_ = "unknown";
    std::string_view file_name_ = "unknown";
    int line_ = 0;
    int column_ = 0;
};

/**
 * @brief Base class for all custom exception types. Forwards its message to [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error)
 *        and saves the call side source location information.
 */
class exception : public std::runtime_error {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message to [`std::runtime_error`](https://en.cppreference.com/w/cpp/error/runtime_error).
     * @param[in] msg the exception's `what()` message
     * @param[in] class_name the name of the thrown exception class
     * @param[in] loc the exception's call side information
     */
    explicit exception(const std::string &msg, const std::string_view class_name = "exception", source_location loc = source_location::current()) :
        std::runtime_error{ msg }, class_name_{ class_name }, loc_{ loc } {}

    /**
     * @brief Returns the information of the call side where the exception was thrown.
     * @return the exception's call side information (`[[nodiscard]]`)
     */
    [[nodiscard]] const source_location &loc() const noexcept { return loc_; }

    /**
     * @brief Returns a sting containing the exception's `what()` message, the name of the thrown exception class and information about the call
     *        side where the exception has been thrown.
     * @return the exception's `what()` message including source location information
     */
    [[nodiscard]] std::string what_with_loc() const {
        return fmt::format(
            "{}\n"
            "{} thrown:\n"
            "  in file      {}\n"
            "  in function  {}\n"
            "  @ line       {}",
            this->what(),
            class_name_,
            loc_.file_name(),
            loc_.function_name(),
            loc_.line());
    }

  private:
    const std::string_view class_name_;
    source_location loc_;
};

/**
 * @brief Exception type thrown if the provided data set file couldn't be found.
 */
class file_not_found_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit file_not_found_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg, "file_not_found_exception", loc } {}
};

/**
 * @brief Exception type thrown if the provided data set file has an invalid format for the selected parser
 *        (e.g. if the arff parser tries to parse a libsvm file).
 */
class invalid_file_format_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit invalid_file_format_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg, "invalid_file_format_exception", loc } {}
};

/**
 * @brief Exception type thrown if the requested backend is not supported on the target machine.
 */
class unsupported_backend_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit unsupported_backend_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg, "unsupported_backend_exception", loc } {}
};

/**
 * @brief Exception type thrown if no data distribution between multiple devices could be created.
 */
class distribution_exception : public exception {
  public:
    /**
     * @brief Construct a new exception forwarding the exception message and source location to `plssvm::exception`.
     * @param[in] msg the exception's `what()` message
     * @param[in] loc the exception's call side information
     */
    explicit distribution_exception(const std::string &msg, source_location loc = source_location::current()) :
        exception{ msg, "distribution_exception", loc } {}
};

}  // namespace plssvm