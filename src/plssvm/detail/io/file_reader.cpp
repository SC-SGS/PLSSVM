/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/io/file_reader.hpp"

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::starts_with, plssvm::detail::trim_left
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::file_not_found_exception, plssvm::invalid_file_format_exception

#include "fmt/core.h"  // fmt::format

// check if memory mapping can be supported
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    #include <fcntl.h>     // open, O_RDONLY
    #include <sys/mman.h>  // mmap, munmap
    #include <sys/stat.h>  // fstat
    #include <unistd.h>    // close
#endif

#include <filesystem>   // std::filesystem::path
#include <fstream>      // std::ifstream
#include <ios>          // std::ios, std::streamsize
#include <iostream>     // std::cerr, std::endl
#include <limits>       // std::numeric_limits
#include <memory>       // std::addressof
#include <string>       // std::string
#include <string_view>  // std::string_view
#include <utility>      // std::exchange, std::move, std::swap
#include <vector>       // std::vector

namespace plssvm::detail::io {

file_reader::file_reader(const char *filename) {
    // open the provided file
    this->open(filename);
}
file_reader::file_reader(const std::string &filename) {
    // open the provided file
    this->open(filename);
}
file_reader::file_reader(const std::filesystem::path &filename) {
    // open the provided file
    this->open(filename);
}

file_reader::~file_reader() {
    // close the file at the end
    this->close();
}

file_reader::file_reader(file_reader &&other) noexcept :
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    file_descriptor_{ std::exchange(other.file_descriptor_, 0) },
    must_unmap_file_{ std::exchange(other.must_unmap_file_, false) },
#endif
    file_content_{ std::exchange(other.file_content_, nullptr) },
    num_bytes_{ std::exchange(other.num_bytes_, 0) },
    lines_{ std::move(other.lines_) },
    is_open_{ std::exchange(other.is_open_, false) } {
}

file_reader &file_reader::operator=(file_reader &&other) noexcept {
    // guard against self assignment
    if (this != std::addressof(other)) {
        // use copy and swap idiom
        file_reader tmp{ std::move(other) };
        this->swap(tmp);
    }
    return *this;
}

void file_reader::open(const char *filename) {
    // no other file might be currently open in this file_reader
    if (this->is_open()) {
        throw file_reader_exception{ "This file_reader is already associated to a file!" };
    }

#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    // headers for memory mapped IO are present -> try it
    this->open_memory_mapped_file(filename);
#else
    // memory mapped IO headers are missing -> use std::ifstream instead
    this->open_file(filename);
#endif

    is_open_ = true;
}
void file_reader::open(const std::string &filename) {
    // open the provided file
    this->open(filename.c_str());
}
void file_reader::open(const std::filesystem::path &filename) {
    // open the provided file
    this->open(filename.string().c_str());
}

bool file_reader::is_open() const noexcept {
    return is_open_;
}
void file_reader::close() {
    if (this->is_open()) {
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
        if (must_unmap_file_) {
            // unmap file
            ::munmap(file_content_, num_bytes_);
            // close file descriptor
            ::close(file_descriptor_);
        }
        file_descriptor_ = 0;
        file_content_ = nullptr;
        must_unmap_file_ = false;
#endif
        // delete allocated buffer (deleting nullptr is a no-op)
        delete[] file_content_;
        file_content_ = nullptr;
        num_bytes_ = 0;

        // clear lines vector
        lines_.clear();

        is_open_ = false;
    }
}

void file_reader::swap(file_reader &other) {
    using std::swap;
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    swap(file_descriptor_, other.file_descriptor_);
    swap(must_unmap_file_, other.must_unmap_file_);
#endif
    swap(file_content_, other.file_content_);
    swap(num_bytes_, other.num_bytes_);
    swap(lines_, other.lines_);
    swap(is_open_, other.is_open_);
}

const std::vector<std::string_view> &file_reader::read_lines(const std::string_view comment) {
    if (!this->is_open()) {
        throw file_reader_exception{ "This file_reader is currently not associated to a file!" };
    }
    // create view from buffer
    std::string_view file_content_view{ file_content_, static_cast<std::string_view::size_type>(num_bytes_) };
    std::string_view::size_type pos = 0;
    while (true) {
        // find newline
        const std::string_view::size_type next_pos = file_content_view.find_first_of('\n', pos);
        if (next_pos == std::string_view::npos) {
            break;
        }
        // remove trailing whitespaces
        const std::string_view sv = trim_left(std::string_view{ file_content_view.data() + pos, next_pos - pos });
        // add line iff the line is not empty and doesn't with a comment
        if (!sv.empty() && !starts_with(sv, comment)) {
            lines_.push_back(sv);
        }
        pos = next_pos + 1;
    }
    // add last line
    const std::string_view sv = trim_left(std::string_view{ file_content_view.data() + pos, file_content_view.size() - pos });
    if (!sv.empty() && !starts_with(sv, comment)) {
        lines_.push_back(sv);
    }

    return lines_;
}
const std::vector<std::string_view> &file_reader::read_lines(const char comment) {
    return this->read_lines(std::string_view{ &comment, 1 });
}

typename std::vector<std::string_view>::size_type file_reader::num_lines() const noexcept {
    return lines_.size();
}
std::string_view file_reader::line(const typename std::vector<std::string_view>::size_type pos) const {
    PLSSVM_ASSERT(pos < this->num_lines(), "Out-of-bounce access!: {} >= {}", pos, this->num_lines());
    return lines_[pos];
}
const std::vector<std::string_view> &file_reader::lines() const noexcept {
    return lines_;
}
const char *file_reader::buffer() const noexcept {
    return file_content_;
}

#if defined(PLSSVM_HAS_MEMORY_MAPPING)
void file_reader::open_memory_mapped_file(const char *filename) {
    // open the file
    file_descriptor_ = ::open(filename, O_RDONLY);
    struct stat attr {};
    // check if file could be opened
    if (fstat(file_descriptor_, &attr) == -1) {
        ::close(file_descriptor_);
        throw file_not_found_exception{ fmt::format("Couldn't find file: '{}'!", filename) };
    }
    if (attr.st_size == 0) {
        // can't memory map empty file
        ::close(file_descriptor_);
        this->open_file(filename);
    } else {
        // memory map file
        file_content_ = static_cast<char *>(mmap(nullptr, attr.st_size, PROT_READ, MAP_SHARED, file_descriptor_, 0));
        // check if memory mapping was successful
        if (static_cast<void *>(file_content_) == MAP_FAILED) {
            // memory mapping wasn't successful -> try reading file with std::ifstream
            ::close(file_descriptor_);
            std::cerr << "Memory mapping failed, falling back to std::ifstream." << std::endl;
            this->open_file(filename);
        } else {
            // set size
            num_bytes_ = static_cast<std::streamsize>(attr.st_size);
            must_unmap_file_ = true;
        }
    }
}
#endif

void file_reader::open_file(const char *filename) {
    // open the file
    std::ifstream f{ filename };
    if (f.fail()) {
        throw file_not_found_exception{ fmt::format("Couldn't find file: '{}'!", filename) };
    }

    // touch all characters in file
    f.ignore(std::numeric_limits<std::streamsize>::max());
    // get number of visited characters
    num_bytes_ = f.gcount();
    // since ignore will have set eof
    f.clear();
    // jump to file start
    f.seekg(0, std::ios_base::beg);

    if (num_bytes_ > 0) {
        // allocate the necessary buffer
        file_content_ = new char[num_bytes_];
        // read the whole file in one go
        if (!f.read(file_content_, num_bytes_)) {
            throw invalid_file_format_exception{ fmt::format("Error while reading file: '{}'!", filename) };
        }
    }
}

void swap(file_reader &lhs, file_reader &rhs) {
    lhs.swap(rhs);
}

}  // namespace plssvm::detail::io