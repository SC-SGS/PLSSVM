// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include "dict_node.hpp"
#include "token.hpp"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

namespace json {

class json : public dict_node {
private:
  std::string fileName;

  std::vector<token> tokenize(const std::string &input);

public:
  explicit json(const std::string &fileName);

  json();

  json(const json &original);

  virtual json *clone();

  void clear();

  void serialize(const std::string &outFileName);

  void deserialize(std::string content);

  void deserializeFromString(const std::string &content);

  using dict_node::serialize;
};

} // namespace json
