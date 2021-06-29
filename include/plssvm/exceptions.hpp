#pragma once

#include <stdexcept>

namespace plssvm {

 class exception : public std::runtime_error {
  public:
    explicit exception(const std::string& msg) : std::runtime_error{msg} { }
 };

 class file_not_found_exception : public exception {
  public:
   explicit file_not_found_exception(const std::string& msg) : exception{msg} { }
 };

 class invalid_file_format_exception : public exception {
  public:
   explicit invalid_file_format_exception(const std::string& msg) : exception{msg} { }
 };

 class unsupported_backend_exception : public exception {
  public:
   explicit unsupported_backend_exception(const std::string& msg) : exception{msg} { }
 };

}