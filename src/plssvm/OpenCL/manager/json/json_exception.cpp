// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "json_exception.hpp"

#include <sstream>
#include <string>

namespace json {

json_exception::json_exception(token &t, const std::string &message) {
  std::stringstream messageStream;
  messageStream << "error: (line: " << t.lineNumber
                << ", char: " << t.charNumber << ") at \"" << t.value
                << "\": ";
  messageStream << message;
  this->message = messageStream.str();
}

json_exception::json_exception(const std::string &message) {
  std::stringstream messageStream;
  messageStream << "error: ";
  messageStream << message;
  this->message = messageStream.str();
}

const char *json_exception::what() const throw() {
  return this->message.c_str();
}

} // namespace json
