// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include "node.hpp"

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace json {

class dict_node : public node {
 protected:
  std::map<std::string, std::unique_ptr<node>> attributes;

  std::vector<std::string> keyOrder;

 public:
  dict_node();

  dict_node(const dict_node &original);

  dict_node &operator=(const dict_node &right);

  node &operator=(const node &right) override;

  void parse(std::vector<token>::iterator &stream_it,
             std::vector<token>::iterator &stream_end) override;

  void parseAttributes(std::vector<token>::iterator &stream_it,
                       std::vector<token>::iterator &stream_end);

  void serialize(std::ostream &outFile, size_t indentWidth) override;

  node &operator[](const std::string &key) override;

  size_t size() override;

  node *clone() override;

  void addAttribute(const std::string &name, std::unique_ptr<node> node) override;

  std::unique_ptr<node> removeAttribute(const std::string name) override;

  // returns the node to which the attribute was added
  node &addTextAttr(const std::string &name, const std::string &value) override;

  // returns the node to which the attribute was added
  node &addIDAttr(const std::string &name, const std::string &value) override;

  // returns the node to which the attribute was added
  // cast internally to string, prevents the boolean overload from being used,
  // if the value is a
  // string literal
  node &addIDAttr(const std::string &name, const char *value) override;

  // returns the node to which the attribute was added
  node &addIDAttr(const std::string &name, const double &value) override;

  // returns the node to which the attribute was added
  node &addIDAttr(const std::string &name, const uint64_t &value) override;

  // returns the node to which the attribute was added
  node &addIDAttr(const std::string &name, const int64_t &value) override;

  // returns the node to which the attribute was added
  node &addIDAttr(const std::string &name, const bool &value) override;

  // returns created dict node
  node &addDictAttr(const std::string &name) override;

  // returns created list node
  node &addListAttr(const std::string &name) override;

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist,
  // the old node is deleted
  node &replaceTextAttr(const std::string &name, const std::string &value) override;

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist,
  // the old node is deleted
  node &replaceIDAttr(const std::string &name, const std::string &value) override;

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist,
  // the old node is deleted
  // cast internally to string, prevents the boolean overload from being used,
  // if the value is a
  // string literal
  node &replaceIDAttr(const std::string &name, const char *value) override;

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist,
  // the old node is deleted
  node &replaceIDAttr(const std::string &name, const double &value) override;

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist,
  // the old node is deleted
  node &replaceIDAttr(const std::string &name, const uint64_t &value) override;

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist,
  // the old node is deleted
  node &replaceIDAttr(const std::string &name, const int64_t &value) override;

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist,
  // the old node is deleted
  node &replaceIDAttr(const std::string &name, const bool &value) override;

  // returns created dict node
  // replaces a node, adds a new node, if the node does not exist,
  // the old node is deleted
  node &replaceDictAttr(const std::string &name) override;

  // returns created list node
  // replaces a node, adds a new node, if the node does not exist,
  // the old node is deleted
  node &replaceListAttr(const std::string &name) override;

  bool contains(const std::string &key) override;

  std::unique_ptr<node> erase(node &node) override;

  std::vector<std::string> &keys() override;
};

}  // namespace json
