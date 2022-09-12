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

#include <filesystem>   // std::filesystem::path
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
    explicit file_reader(const char *filename);
    /**
     * @copydoc plssvm::detail::io::file_reader::file_reader(const char *)
     */
    explicit file_reader(const std::string &filename);
    /**
     * @copydoc plssvm::detail::io::file_reader::file_reader(const char *)
     */
    explicit file_reader(const std::filesystem::path &filename);
    /**
     * @brief Closes the associated file.
     * @details If memory mapped IO has been used, unmap the file and close the file descriptor, and delete the allocated buffer.
     */
    ~file_reader();

    /**
     * @brief Associates the current file_reader with the file denoted by @p filename, i.e., opens the file @p filename (possible memory mapping it).
     * @details This function is called by the constructor of file_reader accepting a std::string and is not usually invoked directly.
     * @param[in] filename the file to open
     * @throws plssvm::file_reader_exception if the file_reader has already opened another file
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     */
    void open(const char *filename);
    /**
     * @copydoc plssvm::detail::io::file_reader::open(const char *)
     */
    void open(const std::string &filename);
    /**
     * @copydoc plssvm::detail::io::file_reader::open(const char *)
     */
    void open(const std::filesystem::path &filename);
    /**
     * @brief Checks whether this file_reader is currently associated with a file.
     * @return `true` if a file is currently open, `false` otherwise (`[[nodiscard]]`)
     */
    [[nodiscard]] bool is_open() const noexcept;
    /**
     * @brief Closes the associated file.
     * @details If memory mapped IO has been used, unmap the file and close the file descriptor, and delete the allocated buffer.
     *          This function is called by the destructor of file_reader when the object goes out of scope and is not usually invoked directly.
     */
    void close();

    /**
     * @brief Read the content of the associated file and splits it into lines, ignoring empty lines and lines starting with the @p comment.
     * @param[in] comment a character (sequence) at the beginning of a line that causes this line to be ignored (used to filter comments)
     * @throws plssvm::file_reader_exception if no file is currently associated to this file_reader
     * @return the split lines, ignoring empty lines and lines starting with the @p comment
     */
    const std::vector<std::string_view> &read_lines(std::string_view comment = { "\n" });
    /**
     * @copydoc plssvm::detail::io::file_reader::read_lines(std::string_view)
     */
    const std::vector<std::string_view> &read_lines(char comment);

    /**
     * @brief Return the number of parsed lines (where all empty lines or lines starting with a comment are ignored).
     * @details Returns `0` if no file is currently associated with this file_reader or the read_kines() function has not been called yet.
     * @return the number of lines after preprocessing (`[[nodiscard]]`)
     */
    [[nodiscard]] typename std::vector<std::string_view>::size_type num_lines() const noexcept;
    /**
     * @brief Return the @p pos line of the parsed file.
     * @details Returns `0` if no file is currently associated with this file_reader or the read_kines() function has not been called yet.
     * @param[in] pos the line to return
     * @return the line without leading whitespaces (`[[nodiscard]]`)
     */
    [[nodiscard]] std::string_view line(typename std::vector<std::string_view>::size_type pos) const;
    /**
     * @brief Return all lines present after the preprocessing.
     * @details Returns `0` if no file is currently associated with this file_reader or the read_kines() function has not been called yet.
     * @return all lines after preprocessing (`[[nodiscard]]`)
     */
    [[nodiscard]] const std::vector<std::string_view> &lines() const noexcept;

  private:
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    /**
     * @brief Try to open the file @p filename and "read" its content using memory mapped IO.
     * @details Currently memory mapped IO is only supported under Linux systems. If the file could not be memory mapped, automatically falls back to open_file().
     * @param[in] filename the file to open
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     */
    void open_memory_mapped_file(const char *filename);
#endif
    /**
     * @brief Read open the file and read its content in one buffer using a normal std::ifstream.
     * @param[in] filename the file to open
     * @throws plssvm::file_not_found_exception if the @p filename couldn't be found
     * @throws plssvm::invalid_file_format_exception if an error occurs while reading the contents of the @p filename
     */
    void open_file(const char *filename);

#if defined(PLSSVM_HAS_MEMORY_MAPPING)
    /// The file descriptor used in the POSIX file and memory mapped IO functions.
    int file_descriptor_{ 0 };
    /// `true` if the file could successfully be memory mapped and, therefore, must also be unmapped at the end, `false` otherwise.
    bool must_unmap_file_{ false };
#endif
    /// The content of the file. Pointer to the memory mapped area or to a separately allocated memory area holding the file's content.
    char *file_content_{ nullptr };
    /// The number of bytes stored in file_content_.
    std::streamsize num_bytes_{ 0 };
    /// The parsed content of file_content_: a vector of all lines that are not empty and do not start with the provided comment.
    std::vector<std::string_view> lines_{};
    /// `true` if a file is currently associated wih this file_reader, `false` otherwise.
    bool is_open_{ false };
};

}  // namespace plssvm::detail::io

#endif  // PLSSVM_DETAIL_IO_FILE_READER_HPP_
