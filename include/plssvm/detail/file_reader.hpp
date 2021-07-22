/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright
 *
 * @brief Implements a file reader class responsible for reading the input file and parsing it into lines.
 */

#pragma once

#include "plssvm/detail/assert.hpp"          // PLSSVM_ASSERT
#include "plssvm/detail/string_utility.hpp"  // plssvm::detail::starts_with, plssvm::detail::trim_left
#include "plssvm/exceptions/exceptions.hpp"  // plssvm::file_not_found_exception

#include "fmt/core.h"  // fmt::format

// check if memory mapping can be supported
#if __has_include(<fcntl.h>) && __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>)
    #include <fcntl.h>     // open, O_RDONLY
    #include <sys/mman.h>  // mmap, munmap
    #include <sys/stat.h>  // fstat
    #include <unistd.h>    // close

    #define PLSSVM_HAS_MEMORY_MAPPING
#endif

#include <cstddef>      // std::size_t
#include <fstream>      // std::ifstream
#include <ios>          // std::ios::beg, std::ios::end, std::streamsize
#include <iostream>     // std::cerr
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm::detail {

/**
 * @brief The `plssvm::detail::file_reader` class is responsible for reading a file and splitting it into its lines.
 * @details If the necessary headers are present, the class tries to memory map the given file. If this fails or if the headers are not present,
 *          the file is read as one blob using [`std::ifstream::read`](https://en.cppreference.com/w/cpp/io/basic_ifstream).
 */
class file_reader {
  public:
    /**
     * @brief Reads the file denoted by @p filename (possibly using memory mapped IO) and splits it into lines, ignoring empty lines and lines starting with
     *        the comment token @p comment.
     * @param[in] filename the file to read and split into lines
     * @param[in] comment the character used to denote comments
     */
    file_reader(const std::string_view filename, const char comment) {
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

    /**
     * @brief Unmap the file and close the file descriptor used by the memory mapped IO operations or delete the allocated buffer.
     */
    ~file_reader() {
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

    /**
     * @brief Return the number of parsed lines.
     * @details All empty lines or lines starting with a comment are ignored.
     * @return the number of lines after preprocessing (`[[nodiscard]]`)
     */
    [[nodiscard]] std::size_t num_lines() const noexcept {
        return lines_.size();
    }
    /**
     * @brief Return the @p pos line of the parsed file.
     * @param[in] pos the line to return
     * @return the line without leading whitespaces
     */
    [[nodiscard]] std::string_view line(const std::size_t pos) const {
        PLSSVM_ASSERT(pos < this->num_lines(), "Out-of-bounce access!: {} >= {}", pos, this->num_lines());
        return lines_[pos];
    }

  private:
    /*
     * Try to read the file using memory mapped IO.
     */
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    void open_memory_mapped_file(const std::string_view filename) {
        // open the file
        file_descriptor_ = open(filename.data(), O_RDONLY);
        struct stat attr {};
        // check if file could be opened
        if (fstat(file_descriptor_, &attr) == -1) {
            close(file_descriptor_);
            throw file_not_found_exception{ fmt::format("Couldn't find file: {}!", filename) };
        }
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
            num_bytes_ = attr.st_size;
            must_unmap_file_ = true;
        }
    }
#endif

    /*
     * Read the file using a normal std::ifstream.
     */
    void open_file(const std::string_view filename) {
        // open the file
        std::ifstream f{ filename.data() };
        if (f.fail()) {
            throw file_not_found_exception{ fmt::format("Couldn't find file: '{}'!", filename) };
        }
        // get the size of the file
        f.seekg(0, std::ios::end);
        num_bytes_ = f.tellg();
        f.seekg(0);
        // allocate the necessary buffer
        file_content_ = new char[num_bytes_];
        // read the whole file in one go
        f.read(file_content_, static_cast<std::streamsize>(num_bytes_));
    }

    /*
     * Split the file into its lines, ignoring empty lines and lines starting with a comment.
     */
    void parse_lines(const char comment) {
        // create view from buffer
        std::string_view file_content_view{ file_content_, num_bytes_ };
        std::size_t pos = 0;
        while (true) {
            // find newline
            std::size_t next_pos = file_content_view.find_first_of('\n', pos);
            if (next_pos == std::string_view::npos) {
                break;
            }
            // remove trailing whitespaces
            std::string_view sv = trim_left(std::string_view{ file_content_view.data() + pos, next_pos - pos });
            // add line iff the line is not empty and doesn't with a comment
            if (!sv.empty() && !starts_with(sv, comment)) {
                lines_.push_back(sv);
            }
            pos = next_pos + 1;
        }
        // add last line
        std::string_view sv = trim_left(std::string_view{ file_content_view.data() + pos, file_content_view.size() - pos });
        if (!sv.empty() && !starts_with(sv, comment)) {
            lines_.push_back(sv);
        }
    }

#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    int file_descriptor_ = 0;
    bool must_unmap_file_ = false;
#endif
    char *file_content_ = nullptr;
    std::size_t num_bytes_ = 0;
    std::vector<std::string_view> lines_{};
};

}  // namespace plssvm::detail

#undef PLSSVM_HAS_MEMORY_MAPPING