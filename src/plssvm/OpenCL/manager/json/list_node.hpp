// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include "node.hpp"

#include <memory>
#include <string>
#include <vector>

namespace json {

class list_node : public node {
 private:
  std::vector<std::unique_ptr<node>> list;

 public:
  list_node();

  list_node(const list_node &original);

  list_node &operator=(const list_node &right);

  node &operator=(const node &right) override;

  void parse(std::vector<token>::iterator &stream_it,
             std::vector<token>::iterator &stream_end) override;

  void serialize(std::ostream &outFile, size_t indentWidth) override;

  node &operator[](const size_t index) override;

  size_t size() override;

  void addValue(std::unique_ptr<node> node) override;

  std::unique_ptr<node> removeValue(size_t index) override;

  node *clone() override;

  // returns created dict node
  node &addDictValue() override;

  // returns created dict node
  node &addListValue() override;

  // returns the list node to which the value was added
  node &addTextValue(const std::string &value) override;

  // returns the list node to which the value was added
  // cast internally to string, prevents the boolean overload from being used,
  // if the value is a
  // string literal
  node &addIdValue(const char *value) override;

  // returns the list node to which the value was added
  node &addIdValue(const std::string &value) override;

  // returns the list node to which the value was added
  node &addIdValue(const double &value) override;

  // returns the list node to which the value was added
  node &addIdValue(const uint64_t &value) override;

  // returns the list node to which the value was added
  node &addIdValue(const int64_t &value) override;

  // returns the list node to which the value was added
  node &addIdValue(const bool &value) override;

  std::unique_ptr<node> erase(node &node) override;
};

}  // namespace json
