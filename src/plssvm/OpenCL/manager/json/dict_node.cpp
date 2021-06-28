// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "dict_node.hpp"

#include "id_node.hpp"
#include "json_exception.hpp"
#include "list_node.hpp"
#include "text_node.hpp"

#include <fstream>
#include <iostream>
#include <string>
#include <vector>

namespace json {

dict_node::dict_node() {}

dict_node::dict_node(const dict_node &original) {
  this->keyOrder = original.keyOrder;
  this->orderedKeyIndex = original.orderedKeyIndex;
  this->parent = nullptr;

  for (auto &tuple : original.attributes) {
    std::unique_ptr<node> clonedValue(tuple.second->clone());
    clonedValue->parent = this;
    this->attributes[tuple.first] = std::move(clonedValue);
  }
}

dict_node &dict_node::operator=(const dict_node &right) {
  this->keyOrder = right.keyOrder;
  this->orderedKeyIndex = right.orderedKeyIndex;
  this->parent = nullptr;

  this->attributes.clear();

  for (auto &tuple : right.attributes) {
    std::unique_ptr<node> clonedValue(tuple.second->clone());
    clonedValue->parent = this;
    this->attributes[tuple.first] = std::move(clonedValue);
  }

  return *this;
}

node &dict_node::operator=(const node &right) {
  const dict_node &dictNode = dynamic_cast<const dict_node &>(right);
  this->operator=(dictNode);
  return *this;
}

void dict_node::parse(std::vector<token>::iterator &stream_it,
                      std::vector<token>::iterator &stream_end) {
  // special case for initial and final brace
  if ((*stream_it).type != token_type::LBRACE) {
    throw json_exception((*stream_it), "expected \"{\"");
  }

  stream_it++;

  // special case for empty dict
  if ((*stream_it).type != token_type::RBRACE) {
    this->parseAttributes(stream_it, stream_end);
  }

  if ((*stream_it).type != token_type::RBRACE) {
    throw json_exception((*stream_it), "expected \"}\"");
  }

  stream_it++;
}

void dict_node::parseAttributes(std::vector<token>::iterator &stream_it,
                                std::vector<token>::iterator &stream_end) {
  //  enum class Rules {
  //    NONE, TEXT_ASSIGN, ID_ASSIGN, LIST_ASSIGN, DICT_ASSIGN
  //  };
  //
  //  Rules rule = Rules::NONE;
  //  size_t i = 0;

  enum class State { NEXT = 0, COLON = 1, VALUE = 2, COMMAFINISH = 3 };

  State state = State::NEXT;

  std::string attributeName;

  // while (stream.size() > 0) {
  // while (stream_it) {
  while (stream_it != stream_end) {
    if (state == State::NEXT) {
      if ((*stream_it).type == token_type::STRING) {
        attributeName = (*stream_it).value;
        state = State::COLON;
        stream_it++;
      } else {
        throw json_exception((*stream_it), "expected attribute key");
      }
    } else if (state == State::COLON) {
      if ((*stream_it).type == token_type::COLON) {
        state = State::VALUE;
        stream_it++;
      } else {
        throw json_exception((*stream_it), "expected \":\"");
      }
    } else if (state == State::VALUE) {
      if ((*stream_it).type == token_type::STRING) {
        auto textNode = std::unique_ptr<text_node>(new text_node());
        textNode->parse(stream_it, stream_end);
        textNode->orderedKeyIndex = this->keyOrder.size();
        this->keyOrder.push_back(attributeName);
        textNode->parent = this;
        this->attributes[attributeName] = std::move(textNode);
        state = State::COMMAFINISH;
      } else if ((*stream_it).type == token_type::ID) {
        auto idNode = std::unique_ptr<id_node>(new id_node());
        idNode->parse(stream_it, stream_end);
        idNode->orderedKeyIndex = this->keyOrder.size();
        this->keyOrder.push_back(attributeName);
        idNode->parent = this;
        this->attributes[attributeName] = std::move(idNode);
        state = State::COMMAFINISH;
      } else if ((*stream_it).type == token_type::LBRACKET) {
        auto listNode = std::unique_ptr<list_node>(new list_node());
        listNode->parse(stream_it, stream_end);
        listNode->orderedKeyIndex = this->keyOrder.size();
        this->keyOrder.push_back(attributeName);
        listNode->parent = this;
        this->attributes[attributeName] = std::move(listNode);
        state = State::COMMAFINISH;
      } else if ((*stream_it).type == token_type::LBRACE) {
        auto attributeNode = std::unique_ptr<dict_node>(new dict_node());
        attributeNode->parse(stream_it, stream_end);
        attributeNode->orderedKeyIndex = this->keyOrder.size();
        this->keyOrder.push_back(attributeName);
        attributeNode->parent = this;
        this->attributes[attributeName] = std::move(attributeNode);
        state = State::COMMAFINISH;
      } else {
        throw json_exception((*stream_it),
                             "expected attribute value type "
                             "(string, id, list or dict)");
      }
    } else if (state == State::COMMAFINISH) {
      if ((*stream_it).type == token_type::COMMA) {
        stream_it++;
        state = State::NEXT;
      } else if ((*stream_it).type == token_type::RBRACE) {
        return;
      } else {
        throw json_exception((*stream_it), "expected \",\" or \"}\"");
      }
    }
  }  // namespace json

  throw json_exception("unexpected end-of-file");
}  // namespace json

node &dict_node::operator[](const std::string &key) {
  if (this->attributes.count(key) == 0) {
    throw json_exception("operator[](): key not found: " + key);
  }

  return *(this->attributes[key]);
}

void dict_node::serialize(std::ostream &outFile, size_t indentWidth) {
  std::string indentation(indentWidth, ' ');
  std::string attrIndentation(indentWidth + node::SERIALIZE_INDENT, ' ');

  outFile << "{" << std::endl;
  bool first = true;

  for (const std::string &key : this->keyOrder) {
    if (first) {
      first = false;
    } else {
      outFile << "," << std::endl;
    }

    outFile << attrIndentation << "\"" << key << "\": ";
    this->attributes[key]->serialize(outFile, indentWidth + node::SERIALIZE_INDENT);
  }

  outFile << std::endl << indentation << "}";
}

size_t dict_node::size() { return this->keyOrder.size(); }

node *dict_node::clone() {
  dict_node *newNode = new dict_node(*this);
  return newNode;
}

void dict_node::addAttribute(const std::string &name, std::unique_ptr<node> n) {
  if (n->parent != nullptr) {
    throw json_exception("addAttribute(): attribute was already added");
  } else if (this->attributes.count(name) > 0) {
    throw json_exception("addAttribute(): attribute with same name already exists");
  }

  n->parent = this;
  n->orderedKeyIndex = this->keyOrder.size();
  this->attributes[name] = std::move(n);
  this->keyOrder.push_back(name);
}

std::unique_ptr<node> dict_node::removeAttribute(const std::string name) {
  if (this->attributes.count(name) == 0) {
    throw json_exception("removeAttribute(): attribute not found");
  }

  size_t orderedIndex = this->attributes[name]->orderedKeyIndex;
  auto attribute = std::move(this->attributes[name]);
  attribute->orderedKeyIndex = 0;
  attribute->parent = nullptr;
  size_t erased = this->attributes.erase(name);
  if (erased != 1) {
    throw json_exception("removeAttribute(): attribute was not erased");
  }
  this->keyOrder.erase(this->keyOrder.begin() + orderedIndex);
  // fix the ordered indices of the remaining attributes
  for (auto it = this->attributes.begin(); it != this->attributes.end(); it++) {
    node &n = *it->second;
    if (n.orderedKeyIndex > orderedIndex) {
      n.orderedKeyIndex -= 1;
    }
  }
  return attribute;
}

// returns the node to which the attribute was added
node &dict_node::addTextAttr(const std::string &name, const std::string &value) {
  auto textNode = std::unique_ptr<text_node>(new text_node());
  textNode->set(value);
  this->addAttribute(name, std::move(textNode));
  return *this;
}

// returns the node to which the attribute was added
node &dict_node::addIDAttr(const std::string &name, const std::string &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->set(value);
  this->addAttribute(name, std::move(idNode));
  return *this;
}

// returns the node to which the attribute was added
// cast internally to string, prevents the boolean overload from being used, if
// the value is a
// string literal
node &dict_node::addIDAttr(const std::string &name, const char *value) {
  return this->addIDAttr(name, std::string(value));
}

// returns the node to which the attribute was added
node &dict_node::addIDAttr(const std::string &name, const double &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->setDouble(value);
  this->addAttribute(name, std::move(idNode));
  return *this;
}

// returns the node to which the attribute was added
node &dict_node::addIDAttr(const std::string &name, const uint64_t &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->setUInt(value);
  this->addAttribute(name, std::move(idNode));
  return *this;
}

// returns the node to which the attribute was added
node &dict_node::addIDAttr(const std::string &name, const int64_t &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->setInt(value);
  this->addAttribute(name, std::move(idNode));
  return *this;
}

// returns the node to which the attribute was added
node &dict_node::addIDAttr(const std::string &name, const bool &value) {
  auto idNode = std::unique_ptr<id_node>(new id_node());
  idNode->setBool(value);
  this->addAttribute(name, std::move(idNode));
  return *this;
}

// returns created dict node
node &dict_node::addDictAttr(const std::string &name) {
  auto dictNode = std::unique_ptr<dict_node>(new dict_node());
  auto &reference = *dictNode;  // because dictNode will be invalidated
  this->addAttribute(name, std::move(dictNode));
  return reference;
}

// returns created list node
node &dict_node::addListAttr(const std::string &name) {
  auto listNode = std::unique_ptr<list_node>(new list_node());
  auto &reference = *listNode;  // because listNode will be invalidated
  this->addAttribute(name, std::move(listNode));
  return reference;
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist,
// the old node is deleted
node &dict_node::replaceTextAttr(const std::string &name, const std::string &value) {
  if (this->attributes.count(name) > 0) {
    this->removeAttribute(name);
  }

  this->addTextAttr(name, value);
  return *this;
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist,
// the old node is deleted
node &dict_node::replaceIDAttr(const std::string &name, const std::string &value) {
  if (this->attributes.count(name) > 0) {
    this->removeAttribute(name);
  }

  this->addIDAttr(name, value);
  return *this;
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist,
// the old node is deleted
// cast internally to string, prevents the boolean overload from being used, if
// the value is a
// string literal
node &dict_node::replaceIDAttr(const std::string &name, const char *value) {
  return this->replaceIDAttr(name, std::string(value));
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist,
// the old node is deleted
node &dict_node::replaceIDAttr(const std::string &name, const double &value) {
  if (this->attributes.count(name) > 0) {
    this->removeAttribute(name);
  }

  this->addIDAttr(name, value);
  return *this;
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist,
// the old node is deleted
node &dict_node::replaceIDAttr(const std::string &name, const uint64_t &value) {
  if (this->attributes.count(name) > 0) {
    this->removeAttribute(name);
  }

  this->addIDAttr(name, value);
  return *this;
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist,
// the old node is deleted
node &dict_node::replaceIDAttr(const std::string &name, const int64_t &value) {
  if (this->attributes.count(name) > 0) {
    this->removeAttribute(name);
  }

  this->addIDAttr(name, value);
  return *this;
}

// returns the node to which the attribute was added
// replaces a node, adds a new node, if the node does not exist,
// the old node is deleted
node &dict_node::replaceIDAttr(const std::string &name, const bool &value) {
  if (this->attributes.count(name) > 0) {
    this->removeAttribute(name);
  }

  this->addIDAttr(name, value);
  return *this;
}

// returns created dict node
// replaces a node, adds a new node, if the node does not exist,
// the old node is deleted
node &dict_node::replaceDictAttr(const std::string &name) {
  if (this->attributes.count(name) > 0) {
    this->removeAttribute(name);
  }

  auto &newNode = this->addDictAttr(name);
  return newNode;
}

// returns created list node
// replaces a node, adds a new node, if the node does not exist,
// the old node is deleted
node &dict_node::replaceListAttr(const std::string &name) {
  if (this->attributes.count(name) > 0) {
    this->removeAttribute(name);
  }

  auto &newNode = this->addListAttr(name);
  return newNode;
}

bool dict_node::contains(const std::string &key) {
  if (this->attributes.count(key) > 0) {
    return true;
  } else {
    return false;
  }
}

std::unique_ptr<node> dict_node::erase(node &n) {
  for (auto it = this->attributes.begin(); it != this->attributes.end(); it++) {
    if (it->second.get() == &n) {
      auto temporary = this->removeAttribute(it->first);
      return temporary;
    }
  }

  throw json_exception("erase(node): node not found");
}

std::vector<std::string> &dict_node::keys() { return this->keyOrder; }

}  // namespace json
