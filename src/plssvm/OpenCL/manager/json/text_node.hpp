// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include "node.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace json {

class text_node : public node {
 private:
  std::string value;

  bool isDouble;
  double doubleValue;  // only used for number types
  bool isUnsigned;
  uint64_t unsignedValue;
  bool isSigned;
  int64_t signedValue;
  bool isBool;
  bool boolValue;

  void setupInternalType();

 public:
  text_node();

  text_node& operator=(const text_node& right) = default;

  node& operator=(const node& right) override;

  void parse(std::vector<token>::iterator& stream_it,
             std::vector<token>::iterator& stream_end) override;

  void serialize(std::ostream& outFile, size_t indentWidth) override;

  std::string& get() override;

  void set(const std::string& value) override;

  double getDouble() override;

  void setDouble(double numericValue) override;

  uint64_t getUInt() override;

  void setUInt(uint64_t uintValue) override;

  int64_t getInt() override;

  void setInt(int64_t intValue) override;

  bool getBool() override;

  void setBool(bool boolValue) override;

  size_t size() override;

  node* clone() override;
};

}  // namespace json
