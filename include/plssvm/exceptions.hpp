#pragma once

#include <stdexcept>

namespace plssvm {

class source_location {
  public:
    static source_location current(
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

    [[nodiscard]] std::string_view function_name() const noexcept { return function_name_; }
    [[nodiscard]] std::string_view file_name() const noexcept { return file_name_; }
    [[nodiscard]] int line() const noexcept { return line_; }
    [[nodiscard]] int column() const noexcept { return column_; }

    friend std::ostream &operator<<(std::ostream &out, const source_location &loc) {
        out << "Exception thrown:" << '\n';
        out << "  in file      " << loc.file_name() << '\n';
        out << "  in function  " << loc.function_name() << '\n';
        out << "  @ line       " << loc.line() << '\n';
        return out;
    }

  private:
    std::string_view function_name_ = "unknown";
    std::string_view file_name_ = "unknown";
    int line_ = 0;
    int column_ = 0;
};

class exception : public std::runtime_error {
  public:
    explicit exception(const std::string &msg, source_location loc = source_location::current())
        : std::runtime_error{msg}, loc_{loc} {}

    [[nodiscard]] const source_location &loc() const noexcept { return loc_; }

  private:
    source_location loc_;
};

class file_not_found_exception : public exception {
  public:
    explicit file_not_found_exception(const std::string &msg, source_location loc = source_location::current())
        : exception{msg, loc} {}
};

class invalid_file_format_exception : public exception {
  public:
    explicit invalid_file_format_exception(const std::string &msg, source_location loc = source_location::current())
        : exception{msg, loc} {}
};

class unsupported_backend_exception : public exception {
  public:
    explicit unsupported_backend_exception(const std::string &msg, source_location loc = source_location::current())
        : exception{msg, loc} {}
};

} // namespace plssvm