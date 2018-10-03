// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "list_node.hpp"
#include "dict_node.hpp"
#include "id_node.hpp"
#include "json_exception.hpp"
#include "text_node.hpp"

#include <fstream>
#include <string>
#include <vector>

namespace json {

list_node::list_node() : list() {}

list_node::list_node(const list_node &original) {
  for (auto &element : original.list) {
    std::unique_ptr<node> cloned(element->clone());
    cloned->parent = this;
    this->list.push_back(std::move(cloned));
  }

  this->orderedKeyIndex = original.orderedKeyIndex;
  this->parent = nullptr;
}

list_node &list_node::operator=(const list_node &right) {
  this->list.clear();

  for (auto &element : right.list) {
    std::unique_ptr<node> cloned(element->clone());
    cloned->parent = this;
    this->list.push_back(std::move(cloned));
  }

  this->orderedKeyIndex = right.orderedKeyIndex;
  this->parent = nullptr;
  return *this;
}

node &list_node::operator=(const node &right) {
  const list_node &listNode = dynamic_cast<const list_node &>(right);
  this->operator=(listNode);
  return *this;
}

void list_node::parse(std::vector<token>::iterator &stream_it,
                      std::vector<token>::iterator &stream_end) {
  list.clear();

  enum class State { ITEMVALUE, NEXT };

  if ((*stream_it).type != token_type::LBRACKET) {
    throw json_exception("expected \"[\"");
  }

  // stream_it++;
  stream_it++;

  // special case for empty list
  if ((*stream_it).type == token_type::RBRACKET) {
    // stream_it++;
    stream_it++;
    return;
  }

  State state = State::ITEMVALUE;

  while (stream_it != stream_end) {
    if (state == State::ITEMVALUE) {
      if ((*stream_it).type == token_type::STRING) {
        auto textNode = std::unique_ptr<text_node>(new text_node());
        textNode->parse(stream_it, stream_end);
        textNode->parent = this;
        this->list.push_back(std::move(textNode));
        state = State::NEXT;
      } else if ((*stream_it).type == token_type::ID) {
        auto idNode = std::unique_ptr<id_node>(new id_node());
        idNode->parse(stream_it, stream_end);
        idNode->parent = this;
        this->list.push_back(std::move(idNode));
        state = State::NEXT;
      } else if ((*stream_it).type == token_type::LBRACKET) {
        auto listNode = std::unique_ptr<list_node>(new list_node());
        listNode->parse(stream_it, stream_end);
        listNode->parent = this;
        this->list.push_back(std::move(listNode));
        state = State::NEXT;
      } else if ((*stream_it).type == token_type::LBRACE) {
        auto dictNode = std::unique_ptr<dict_node>(new dict_node());
        dictNode->parse(stream_it, stream_end);
        dictNode->parent = this;
        this->list.push_back(std::move(dictNode));
        state = State::NEXT;
      } else {
        throw json_exception((*stream_it), "expected list value type (string, id, list or dict)");
      }
    } else if (state == State::NEXT) {
      if ((*stream_it).type == token_type::COMMA) {
        stream_it++;
        state = State::ITEMVALUE;
      } else if ((*stream_it).type == token_type::RBRACKET) {
        stream_it++;
        return;
      } else {
        throw json_exception((*stream_it), "expected \",\" or \"]\"");
      }
    }
  }

  throw json_exception("unexpected end-of-file");
}

node &list_node::operator[](size_t index) { return *this->list[index]; }

void list_node::serialize(std::ostream &outFile, size_t indentWidth) {
  outFile << "[";
  bool first = true;

  for (const std::unique_ptr<node> &n : this->list) {
    if (first) {
      first = false;
    } else {
      outFile << ", ";
    }

    n->serialize(outFile, indentWidth + node::SERIALIZE_INDENT);
  }

  outFile << "]";
}

size_t list_node::size() { return this->list.size(); }

void list_node::addValue(std::unique_ptr<node> n) {
  if (n->parent != nullptr) {
    throw json_exception("addItem(): value was already added");
  }

  n->parent = this;
  this->list.push_back(std::move(n));
}

node *list_node::clone() {
  list_node *newNode = new list_node(*this);
  return newNode;
}

std::unique_ptr<node> list_node::removeValue(size_t index) {
  if (index >= list.size()) {
    throw json_exception("removeItem(): index is out-of-bounds");
  }

  auto n = std::move(this->list[index]);
  n->parent = nullptr;
  this->list.erase(this->list.begin() + index);
  return n;
}

// returns the list node to which the value was added
node &list_node::addTextValue(const std::string &value) {
  auto textNode = std::unique_ptr<text_node>(new text_node());
  textNode->set(value);
  this->addValue(std::move(textNode));
  return *this;
}

// returns the list node to which the value was added
node &list_node::addIdValue(const std::string &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->set(value);
  this->addValue(std::move(idNode));
  return *this;
}

// returns the list node to which the value was added
// cast internally to string, prevents the boolean overload from being used, if
// the value is a
// string literal
node &list_node::addIdValue(const char *value) { return this->addIdValue(std::string(value)); }

// returns the list node to which the value was added
node &list_node::addIdValue(const double &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->setDouble(value);
  this->addValue(std::move(idNode));
  return *this;
}

// returns the list node to which the value was added
node &list_node::addIdValue(const uint64_t &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->setUInt(value);
  this->addValue(std::move(idNode));
  return *this;
}

// returns the list node to which the value was added
node &list_node::addIdValue(const int64_t &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->setInt(value);
  this->addValue(std::move(idNode));
  return *this;
}

// returns the list node to which the value was added
node &list_node::addIdValue(const bool &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->setBool(value);
  this->addValue(std::move(idNode));
  return *this;
}

// returns created dict node
node &list_node::addDictValue() {
  auto dictNode = std::unique_ptr<dict_node>(new dict_node());
  auto &reference = *dictNode;  // because dictNode will be invalidated
  this->addValue(std::move(dictNode));
  return reference;
}

// returns created dict node
node &list_node::addListValue() {
  auto listNode = std::unique_ptr<list_node>(new list_node());
  auto &reference = *listNode;  // because listNode will be invalidated
  this->addValue(std::move(listNode));
  return reference;
}

std::unique_ptr<node> list_node::erase(node &n) {
  for (auto it = this->list.begin(); it != this->list.end(); it++) {
    if ((*it).get() == &n) {
      auto n = std::move(*it);
      n->parent = nullptr;
      this->list.erase(it);
      return n;
    }
  }

  throw json_exception("erase(node): node not found");
}

}  // namespace json
