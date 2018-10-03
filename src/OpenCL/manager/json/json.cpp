// Copyright (C) 2008-today The SG++ project
// This file is part of the SG++ project. For conditions of distribution and
// use, please see the copyright notice provided with SG++ or at
// sgpp.sparsegrids.org

#include "json.hpp"
#include "json_exception.hpp"

#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace json {

json::json() : fileName("") {}

json::json(const std::string &fileName) : fileName(fileName) {
  std::ifstream file;
  file.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  try {
    file.open(fileName);
  } catch (std::ifstream::failure &e) {
    std::stringstream stream;
    stream << "json error: could not open file: " << fileName << std::endl;
    throw json_exception(stream.str());
  }

  std::string content;
  try {
    content.assign(std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>());
  } catch (std::ifstream::failure &e) {
    std::stringstream stream;
    stream << "json error: could not successfully read file: " << fileName << std::endl;
    throw json_exception(stream.str());
  }

  file.close();

  std::vector<token> tokenStream = this->tokenize(content);
  auto stream_it = tokenStream.begin();
  auto stream_end = tokenStream.end();
  this->parse(stream_it, stream_end);

  if (stream_it != stream_end) {
    throw json_exception(tokenStream[0], "expected end-of-file");
  }
}

json::json(const json &original) : dict_node(original) { this->fileName = original.fileName; }

std::vector<token> json::tokenize(const std::string &input) {
  std::vector<token> stream;

  token_type state = token_type::NONE;

  token t;
  size_t lineNumber = 1;
  size_t charNumber = 0;

  for (size_t i = 0; i < input.size(); i++) {
    if (input[i] == '\n') {
      lineNumber += 1;
      charNumber = 0;
    } else {
      charNumber += 1;
    }

    // skip whitespace while not tokenizing anything
    if (state == token_type::NONE) {
      if (input[i] == ' ' || input[i] == '\r' || input[i] == '\n' || input[i] == '\t') {
        continue;
      }
    }

    if (state == token_type::NONE) {
      if (input[i] == '{') {
        t.type = token_type::LBRACE;
        t.value = "{";
        t.lineNumber = lineNumber;
        t.charNumber = charNumber;
        stream.push_back(t);
      } else if (input[i] == ',') {
        t.type = token_type::COMMA;
        t.value = ",";
        t.lineNumber = lineNumber;
        t.charNumber = charNumber;
        stream.push_back(t);
      } else if (input[i] == '}') {
        t.type = token_type::RBRACE;
        t.value = "}";
        t.lineNumber = lineNumber;
        t.charNumber = charNumber;
        stream.push_back(t);
      } else if (input[i] == '[') {
        t.type = token_type::LBRACKET;
        t.value = "[";
        t.lineNumber = lineNumber;
        t.charNumber = charNumber;
        stream.push_back(t);
      } else if (input[i] == ']') {
        t.type = token_type::RBRACKET;
        t.value = "]";
        t.lineNumber = lineNumber;
        t.charNumber = charNumber;
        stream.push_back(t);
      } else if (input[i] == ':') {
        t.type = token_type::COLON;
        t.value = ":";
        t.lineNumber = lineNumber;
        t.charNumber = charNumber;
        stream.push_back(t);
      } else if (input[i] == '"') {
        t.type = token_type::STRING;
        t.value = "";
        t.lineNumber = lineNumber;
        t.charNumber = charNumber;
        state = token_type::STRING;
      } else if (input[i] == '/') {
        state = token_type::COMMENT;
      } else {
        t.type = token_type::ID;
        t.value = input[i];
        t.lineNumber = lineNumber;
        t.charNumber = charNumber;
        state = token_type::ID;
      }
    } else if (state == token_type::STRING) {
      if (input[i] == '"') {
        stream.push_back(t);
        state = token_type::NONE;
      } else {
        t.value.push_back(input[i]);
      }
    } else if (state == token_type::ID) {
      if (input[i] == '{' || input[i] == '}' || input[i] == '[' || input[i] == ']' ||
          input[i] == ':' || input[i] == ',' || input[i] == ' ' || input[i] == '\t' ||
          input[i] == '\r' || input[i] == '\n') {
        stream.push_back(t);
        state = token_type::NONE;

        if (input[i] == '\n') {
          lineNumber -= 1;  // as the char will be reprocessed
        }

        i -= 1;  // revert by one

      } else {
        t.value.push_back(input[i]);
      }
    } else if (state == token_type::COMMENT) {
      if (input[i] == '/') {
        state = token_type::SINGLELINE;
      } else if (input[i] == '*') {
        state = token_type::MULTILINECOMMENT;
      } else {
        std::stringstream messageStream;
        messageStream << "error: (line: " << lineNumber << ", char: " << (charNumber - 1)
                      << "): expected a single- or multiline comment after \"/\"";
        throw json_exception(messageStream.str());
      }
    } else if (state == token_type::SINGLELINE) {
      if (input[i] == '\n') {
        state = token_type::NONE;
      }
    } else if (state == token_type::MULTILINECOMMENT) {
      if (input[i] == '*') {
        state = token_type::MULTILINECOMMENTSTAR;
      }
    } else if (state == token_type::MULTILINECOMMENTSTAR) {
      if (input[i] == '/') {
        state = token_type::NONE;
      }
    } else {
      std::stringstream messageStream;
      messageStream << "error: (line: " << lineNumber << ", char: " << (charNumber - 1)
                    << "): illegal parser state, this might be a bug";
      throw json_exception(messageStream.str());
    }
  }

  return stream;
}

void json::serialize(const std::string &outFileName) {
  std::ofstream outFile(outFileName);
  outFile.exceptions(std::ifstream::failbit | std::ifstream::badbit);

  this->serialize(outFile, 0);

  outFile.close();
}

void json::deserialize(std::string content) {
  this->clear();
  std::vector<token> tokenStream = this->tokenize(content);

  auto stream_it = tokenStream.begin();
  auto stream_end = tokenStream.end();
  this->parse(stream_it, stream_end);

  if (tokenStream.size() != 0) {
    throw json_exception(tokenStream[0], "expected end-of-file");
  }
  fileName = "";
}
json *json::clone() { return new json(*this); }

void json::clear() {
  this->fileName = "";
  this->attributes.clear();
  this->keyOrder.clear();
  // it shouldn't even be necessary to reset these
  this->orderedKeyIndex = 0;
  this->parent = nullptr;
}

void json::deserializeFromString(const std::string &content) {
  this->clear();
  std::vector<token> tokenStream = this->tokenize(content);

  auto stream_it = tokenStream.begin();
  auto stream_end = tokenStream.end();
  this->parse(stream_it, stream_end);

  if (tokenStream.size() != 0) {
    throw json_exception(tokenStream[0], "expected end-of-file");
  }
}

}  // namespace json
