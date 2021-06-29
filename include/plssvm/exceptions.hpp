#pragma once

#include <stdexcept>
#include <string_view>

namespace plssvm {

 class exception : public std::runtime_error {
  public:
    explicit exception(const std::string_view msg) : std::runtime_error{msg.data()} { }
 };

 class file_not_found_exception : public exception {
  public:
   explicit file_not_found_exception(const std::string_view msg) : exception{msg} { }
 };

 class invalid_file_format_exception : public exception {
  public:
   explicit invalid_file_format_exception(const std::string_view msg) : excpetion{msg} { }
 };

 class unsupported_backend_exception : public exception {
  public:
   explicit unsupported_backend_exception(const std::string_view msg) : exception{msg} { }
 };

}