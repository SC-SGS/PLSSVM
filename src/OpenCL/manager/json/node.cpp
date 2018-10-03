// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "node.hpp"
#include "json_exception.hpp"

#include <string>
#include <vector>

namespace json {

node::node() : orderedKeyIndex(0), parent(nullptr) {}

node &node::operator=(const node &right) {
  // members of this base class are actually copied
  // in the dervied classes, therefore nothing to do
  return *this;
}

node &node::operator[](const std::string &key) {
  throw json_exception("operator[] is only implemented for dict nodes");
}

node &node::operator[](const size_t index) {
  throw json_exception("operator[] is only implemented for list nodes");
}

std::string &node::get() {
  throw json_exception(
      "getValue() is only implemented for string and id nodes");
}

void node::set(const std::string &value) {
  throw json_exception(
      "setValue() is only implemented for string and id nodes");
}

double node::getDouble() {
  throw json_exception("getNumericValue() is only implemented for id nodes");
}

void node::setDouble(double doubleValue) {
  throw json_exception("setNumericValue() is only implemented for id nodes");
}

uint64_t node::getUInt() {
  throw json_exception("getUInt() is only implemented for id nodes");
}

void node::setUInt(uint64_t uintValue) {
  throw json_exception("setUInt() is only implemented for id nodes");
}

int64_t node::getInt() {
  throw json_exception("getInt() is only implemented for id nodes");
}

void node::setInt(int64_t intValue) {
  throw json_exception("setInt() is only implemented for id nodes");
}

bool node::getBool() {
  throw json_exception("getBool() is only implemented for id nodes");
}

void node::setBool(bool boolValue) {
  throw json_exception("setBool() is only implemented for id nodes");
}

void node::addValue(std::unique_ptr<node> node) {
  throw json_exception("addItem() is only implemented for list nodes");
}

void node::addAttribute(const std::string &name, std::unique_ptr<node> node) {
  throw json_exception("addAttribute() is only implemented for dict nodes");
}

std::unique_ptr<node> node::removeValue(size_t index) {
  throw json_exception(
      "removeItem() is only implemented for attribute and list nodes");
}

std::unique_ptr<node> node::removeAttribute(const std::string name) {
  throw json_exception("removeAttribute() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
node &node::addTextAttr(const std::string &name, const std::string &value) {
  throw json_exception("addTextAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
node &node::addIDAttr(const std::string &name, const std::string &value) {
  throw json_exception("addIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
node &node::addIDAttr(const std::string &name, const char *value) {
  throw json_exception("addIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
node &node::addIDAttr(const std::string &name, const double &numericValue) {
  throw json_exception("addIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
node &node::addIDAttr(const std::string &name, const uint64_t &value) {
  throw json_exception("addIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
node &node::addIDAttr(const std::string &name, const int64_t &value) {
  throw json_exception("addIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
node &node::addIDAttr(const std::string &name, const bool &value) {
  throw json_exception("addIDAttr() is only implemented for dict nodes");
}

// returns created dict node
node &node::addDictAttr(const std::string &name) {
  throw json_exception("addDictAttr() is only implemented for dict nodes");
}

// returns created list node
node &node::addListAttr(const std::string &name) {
  throw json_exception("addListAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist, the old node is
// deleted
node &node::replaceTextAttr(const std::string &name, const std::string &value) {
  throw json_exception("replaceTextAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist, the old node is
// deleted
node &node::replaceIDAttr(const std::string &name, const std::string &value) {
  throw json_exception("replaceIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist, the old node is
// deleted
// cast internally to string, prevents the boolean overload from being used, if
// the value is a
// string literal
node &node::replaceIDAttr(const std::string &name, const char *value) {
  throw json_exception("replaceIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist, the old node is
// deleted
node &node::replaceIDAttr(const std::string &name, const double &value) {
  throw json_exception("replaceIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist, the old node is
// deleted
node &node::replaceIDAttr(const std::string &name, const uint64_t &value) {
  throw json_exception("replaceIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist, the old node is
// deleted
node &node::replaceIDAttr(const std::string &name, const int64_t &value) {
  throw json_exception("replaceIDAttr() is only implemented for dict nodes");
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist, the old node is
// deleted
node &node::replaceIDAttr(const std::string &name, const bool &value) {
  throw json_exception("replaceIDAttr() is only implemented for dict nodes");
}

// returns created dict node
// replaces a node, adds a new node, if the node does not exist, the old node is
// deleted
node &node::replaceDictAttr(const std::string &name) {
  throw json_exception("replaceDictAttr() is only implemented for dict nodes");
}

// returns created list node
// replaces a node, adds a new node, if the node does not exist, the old node is
// deleted
node &node::replaceListAttr(const std::string &name) {
  throw json_exception("replaceListAttr() is only implemented for dict nodes");
}

// returns created dict node
node &node::addDictValue() {
  throw json_exception("addDictValue() is only implemented for list nodes");
}

// returns created dict node
node &node::addListValue() {
  throw json_exception("addListValue() is only implemented for list nodes");
}

// returns the list node to which the value was added
node &node::addTextValue(const std::string &value) {
  throw json_exception("addTextValue() is only implemented for list nodes");
}

// returns the list node to which the value was added
node &node::addIdValue(const std::string &value) {
  throw json_exception("addIdValue() is only implemented for list nodes");
}

// returns the list node to which the value was added
// cast internally to string, prevents the boolean overload from being used, if
// the value is a
// string literal
node &node::addIdValue(const char *value) {
  throw json_exception("addIdValue() is only implemented for list nodes");
}

// returns the list node to which the value was added
node &node::addIdValue(const double &value) {
  throw json_exception("addIdValue() is only implemented for list nodes");
}

// returns the list node to which the value was added
node &node::addIdValue(const uint64_t &value) {
  throw json_exception("addIdValue() is only implemented for list nodes");
}

// returns the list node to which the value was added
node &node::addIdValue(const int64_t &value) {
  throw json_exception("addIdValue() is only implemented for list nodes");
}

// returns the list node to which the value was added
node &node::addIdValue(const bool &value) {
  throw json_exception("addIdValue() is only implemented for list nodes");
}

bool node::contains(const std::string &key) {
  throw json_exception("contains() is only implemented for dict nodes");
}

std::unique_ptr<node> node::erase(node &n) {
  throw json_exception(
      "erase(node) is only implemented for list and dict nodes");
}

std::unique_ptr<node> node::erase() {
  if (this->parent == nullptr) {
    throw json_exception("erase(): has no parent");
  }

  std::unique_ptr<node> self = this->parent->erase(*this);
  this->parent = nullptr;
  return self;
}

std::vector<std::string> &node::keys() {
  throw json_exception("keys() is only implemented for dict nodes");
}

} // namespace json

std::ostream &operator<<(std::ostream &stream, json::node &n) {
  n.serialize(stream, 0);
  return stream;
}
