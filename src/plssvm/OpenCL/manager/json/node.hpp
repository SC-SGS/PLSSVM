// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#pragma once

#include "token.hpp"

#include <memory>
#include <string>
#include <vector>

namespace json {

class node {
 protected:
  static const int SERIALIZE_INDENT = 3;

 public:
  // only relevant if the parent is a dict_node, managed by the dict_node
  size_t orderedKeyIndex;

  // managed by the parent, i.e. dict_node or list_node
  node *parent;

  node();

  virtual ~node() = default;

  virtual node &operator=(const node &right);

  virtual void parse(std::vector<token>::iterator &stream_it,
                     std::vector<token>::iterator &stream_end) = 0;

  virtual void serialize(std::ostream &outFile, size_t indentWidth) = 0;

  virtual node &operator[](const std::string &key);

  virtual node &operator[](const size_t index);

  virtual std::string &get();

  virtual void set(const std::string &value);

  virtual double getDouble();

  virtual void setDouble(double doubleValue);

  virtual uint64_t getUInt();

  virtual void setUInt(uint64_t uintValue);

  virtual int64_t getInt();

  virtual void setInt(int64_t intValue);

  virtual bool getBool();

  virtual void setBool(bool boolValue);

  //  virtual JSONNode &getItem(size_t index);

  virtual size_t size() = 0;

  virtual void addValue(std::unique_ptr<node> node);

  virtual void addAttribute(const std::string &name, std::unique_ptr<node> n);

  virtual std::unique_ptr<node> removeValue(size_t index);

  virtual std::unique_ptr<node> removeAttribute(const std::string name);

  virtual node *clone() = 0;

  // returns the node to which the attribute was added
  virtual node &addTextAttr(const std::string &name, const std::string &value);

  // returns the node to which the attribute was added
  virtual node &addIDAttr(const std::string &name, const std::string &value);

  // returns the node to which the attribute was added
  // cast internally to string, prevents the boolean overload from being used,
  // if the value is a
  // string literal
  virtual node &addIDAttr(const std::string &name, const char *value);

  // returns the node to which the attribute was added
  virtual node &addIDAttr(const std::string &name, const double &value);

  // returns the node to which the attribute was added
  virtual node &addIDAttr(const std::string &name, const uint64_t &value);

  // returns the node to which the attribute was added
  virtual node &addIDAttr(const std::string &name, const int64_t &value);

  // returns the node to which the attribute was added
  virtual node &addIDAttr(const std::string &name, const bool &value);

  // returns created dict node
  virtual node &addDictAttr(const std::string &name);

  // returns created list node
  virtual node &addListAttr(const std::string &name);

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist, the old node
  // is deleted
  virtual node &replaceTextAttr(const std::string &name, const std::string &value);

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist, the old node
  // is deleted
  virtual node &replaceIDAttr(const std::string &name, const std::string &value);

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist, the old node
  // is deleted
  // cast internally to string, prevents the boolean overload from being used,
  // if the value is a
  // string literal
  virtual node &replaceIDAttr(const std::string &name, const char *value);

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist, the old node
  // is deleted
  virtual node &replaceIDAttr(const std::string &name, const double &value);

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist, the old node
  // is deleted
  virtual node &replaceIDAttr(const std::string &name, const uint64_t &value);

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist, the old node
  // is deleted
  virtual node &replaceIDAttr(const std::string &name, const int64_t &value);

  // returns the node to which the attribute was added
  // replaces a node, adds a new node, if the node does not exist, the old node
  // is deleted
  virtual node &replaceIDAttr(const std::string &name, const bool &value);

  // returns created dict node
  // replaces a node, adds a new node, if the node does not exist, the old node
  // is deleted
  virtual node &replaceDictAttr(const std::string &name);

  // returns created list node
  // replaces a node, adds a new node, if the node does not exist, the old node
  // is deleted
  virtual node &replaceListAttr(const std::string &name);

  // returns created dict node
  virtual node &addDictValue();

  // returns created dict node
  virtual node &addListValue();

  // returns the list node to which the value was added
  virtual node &addTextValue(const std::string &value);

  // returns the list node to which the value was added
  virtual node &addIdValue(const std::string &value);

  // returns the list node to which the value was added
  // cast internally to string, prevents the boolean overload from being used,
  // if the value is a
  // string literal
  virtual node &addIdValue(const char *value);

  // returns the list node to which the value was added
  virtual node &addIdValue(const double &value);

  // returns the list node to which the value was added
  virtual node &addIdValue(const uint64_t &value);

  // returns the list node to which the value was added
  virtual node &addIdValue(const int64_t &value);

  // returns the list node to which the value was added
  virtual node &addIdValue(const bool &value);

  virtual bool contains(const std::string &key);

  virtual std::unique_ptr<node> erase(node &n);

  virtual std::unique_ptr<node> erase();

  virtual std::vector<std::string> &keys();
};

}  // namespace json

std::ostream &operator<<(std::ostream &stream, json::node &n);
