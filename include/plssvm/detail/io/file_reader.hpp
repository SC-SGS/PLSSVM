/**
 * @file
 * @author Alexander Van Craen
 * @author Marcel Breyer
 * @copyright 2018-today The PLSSVM project - All Rights Reserved
 * @license This file is part of the PLSSVM project which is released under the MIT license.
 *          See the LICENSE.md file in the project root for full license information.
 *
 * @brief Implements a file reader class responsible for reading the input file and parsing it into lines.
 */

#ifndef PLSSVM_DETAIL_IO_FILE_READER_HPP_
#define PLSSVM_DETAIL_IO_FILE_READER_HPP_
#pragma once

// check if memory mapping can be supported
#if __has_include(<fcntl.h>) && __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>)
    #define PLSSVM_HAS_MEMORY_MAPPING
#endif

#include <ios>          // std::streamsize
#include <string_view>  // std::string_view
#include <vector>       // std::vector

namespace plssvm::detail::io {

/**
 * @brief The plssvm::detail::file_reader class is responsible for reading a file and splitting it into its lines.
 * @details If the necessary headers are present, the class tries to memory map the given file. If this fails or if the headers are not present,
 *          the file is read as one blob using [`std::ifstream::read`](https://en.cppreference.com/w/cpp/io/basic_ifstream).
 */
class file_reader {
  public:
    /**
     * @brief Creates a new file_reader **without** associating it to a file.
     */
    file_reader() = default;
    /**
     * @brief Create a new file_reader and associate it to the @p filename by opening it (possibly memory mapping it).
     * @param[in] filename the file to open
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     */
    explicit file_reader(const std::string &filename);
    /**
     * @brief Unmap the file and close the file descriptor used by the memory mapped IO operations and delete the allocated buffer.
     */
    ~file_reader();

    /**
     * @brief Associates the current file_reader with the file denoted by @p filename, i.e., opens the file @p filename (possible memory mapping it).
     * @details This function is called by the constructor of file_reader accepting a std::string and is not usually invoked directly.
     * @param[in] filename the file to open
     * @throws plssvm::file_reader_exception if the file_reader has already opened another file
     */
    void open(const std::string &filename);
    /**
     * @brief Checks whether this file_reader is currently associated with a file.
     * @return `true` if a file is currently open, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_open() const noexcept;
    /**
     * @brief Unmap the file and close the file descriptor used by the memory mapped IO operations and delete the allocated buffer.
     * @details This function is called by the destructor of file_reader when the object goes out of scope and is not usually invoked directly.
     */
    void close();

    /**
     * @brief Read the content of the associated file and splits it into lines, ignoring empty lines and lines starting with
     *        the comment token @p comment.
     * @param[in] comment a character at the beginning of a line that causes this line to be ignored (used to filter comments)
     * @throws plssvm::file_reader_exception if no file is currently opened by this file_reader
     * @return the split lines
     */
    const std::vector<std::string_view> &read_lines(char comment = '\0');

    /**
     * @brief Return the number of parsed lines.
     * @details All empty lines or lines starting with a comment are ignored.
     * @return the number of lines after preprocessing (`[[nodiscard]]`)
     */
    [[nodiscard]] typename std::vector<std::string_view>::size_type num_lines() const noexcept;
    /**
     * @brief Return the @p pos line of the parsed file.
     * @param[in] pos the line to return
     * @return the line without leading whitespaces (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string_view line(typename std::vector<std::string_view>::size_type pos) const;
    /**
     * @brief Return all lines present after the preprocessing.
     * @return all lines after preprocessing (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<std::string_view> &lines() const noexcept;

  private:
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    /*
     * Try to read the file using memory mapped IO.
     */
    void open_memory_mapped_file(std::string_view filename);
#endif
    /*
     * Read the file using a normal std::ifstream.
     */
    void open_file(std::string_view filename);

#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    int file_descriptor_{ 0 };
    bool must_unmap_file_{ false };
#endif
    char *file_content_{ nullptr };
    std::streamsize num_bytes_{ 0 };
    std::vector<std::string_view> lines_{};
    bool is_open_{ false };
};

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_FILE_READER_HPP_
