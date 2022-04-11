/**
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 */

#include "plssvm/detail/file_reader.hpp"

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

#include <fstream>      // std::ifstream
#include <ios>          // std::ios, std::streamsize
#include <iostream>     // std::cerr, std::endl
#include <limits>       // std::numeric_limits
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm::detail {

file_reader::file_reader(const std::string &filename, const char comment) {
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    // headers for memory mapped IO are present -> try it
    this->open_memory_mapped_file(filename);
#else
    // memory mapped IO headers are missing -> use std::ifstream instead
    this->open_file(filename);
#endif
    // split read data into lines
    this->parse_lines(comment);
}

file_reader::~file_reader() {
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    if (must_unmap_file_) {
        // unmap file
        munmap(file_content_, num_bytes_);
        // close file descriptor
        close(file_descriptor_);
    }
    file_content_ = nullptr;
#endif
    // delete allocated buffer (deleting nullptr is a no-op)
    delete[] file_content_;
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

#if defined(PLSSVM_HAS_MEMORY_MAPPING)
void file_reader::open_memory_mapped_file(const std::string_view filename) {
    // open the file
    file_descriptor_ = open(filename.data(), O_RDONLY);
    struct stat attr {};
    // check if file could be opened
    if (fstat(file_descriptor_, &attr) == -1) {
        close(file_descriptor_);
        throw file_not_found_exception{ fmt::format("Couldn't find file: '{}'!", filename) };
    }
    if (attr.st_size == 0) {
        // can't memory map empty file
        close(file_descriptor_);
        this->open_file(filename);
    } else {
        // memory map file
        file_content_ = static_cast<char *>(mmap(nullptr, attr.st_size, PROT_READ, MAP_SHARED, file_descriptor_, 0));
        // check if memory mapping was successful
        if (static_cast<void *>(file_content_) == MAP_FAILED) {
            // memory mapping wasn't successful -> try reading file with std::ifstream
            close(file_descriptor_);
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

void file_reader::open_file(const std::string_view filename) {
    // open the file
    std::ifstream f{ filename.data() };
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

void file_reader::parse_lines(const char comment) {
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
}

}  // namespace plssvm::detail