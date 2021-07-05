#pragma once

#include <plssvm/exceptions.hpp>
#include <plssvm/string_utility.hpp>

#include <fmt/format.h>

// check if memory mapping can be supported
#if __has_include(<fcntl.h>) && __has_include(<sys/mman.h>) && __has_include(<sys/stat.h>) && __has_include(<unistd.h>)
    #include <fcntl.h>     // open, O_RDONLY
    #include <sys/mman.h>  // mmap
    #include <sys/stat.h>  // fstat
    #include <unistd.h>    // close

    #define PLSSVM_HAS_MEMORY_MAPPING
#endif

#include <fstream>
#include <iostream>
#include <string_view>
#include <vector>

namespace plssvm {

class file {
  public:
    file(const std::string_view filename, const char comment) {
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
        this->open_memory_mapped_file(filename);
#else
        this->open_file(filename);
#endif
        this->parse_lines(comment);
    }
    ~file() {
#if defined(PLSSVM_HAS_MEMORY_MAPPING)
        close(file_descriptor_);
        file_content_ = nullptr;
#endif
        delete[] file_content_;
    }

    [[nodiscard]] std::size_t num_lines() const noexcept {
        return lines_.size();
    }
    [[nodiscard]] std::string_view line(const std::size_t pos) const {
        return lines_[pos];
    }

  private:
    void open_memory_mapped_file(const std::string_view filename) {
        file_descriptor_ = open(filename.data(), O_RDONLY);
        struct stat attr {};
        // check if file could be opened
        if (fstat(file_descriptor_, &attr) == -1) {
            close(file_descriptor_);
            throw file_not_found_exception{ fmt::format("Couldn't find file: {}!", filename) };
        }
        // memory map file
        file_content_ = (char *) mmap(nullptr, attr.st_size, PROT_READ, MAP_SHARED, file_descriptor_, 0);
        // check if memory mapping was successful
        if ((void *) file_content_ == MAP_FAILED) {
            // memory mapping wasn't successful -> try reading file with std::ifstream
            close(file_descriptor_);
            std::cerr << "Memory mapping failed, falling back to std::ifstream." << std::endl;
            this->open_file(filename);
        }
        size_ = attr.st_size;
    }

    void open_file(const std::string_view filename) {
        std::ifstream f{ filename.data() };
        if (f.fail()) {
            throw file_not_found_exception{ fmt::format("Couldn't find file: '{}'!", filename) };
        }
        f.seekg(0, std::ios::end);
        size_ = f.tellg();
        f.seekg(0);
        file_content_ = new char[size_];
        f.read(file_content_, static_cast<std::streamsize>(size_));
    }

    void parse_lines(const char comment) {
        std::string_view file_content_view{ file_content_, size_ };
        std::size_t pos = 0;
        while (true) {
            std::size_t next_pos = file_content_view.find_first_of('\n', pos);
            if (next_pos == std::string::npos) {
                break;
            }
            std::string_view sv = util::trim_left(std::string_view{ file_content_view.data() + pos, next_pos - pos });
            if (!sv.empty() && !util::starts_with(sv, comment)) {
                lines_.push_back(sv);
            }
            pos = next_pos + 1;
        }
        std::string_view sv = util::trim_left(std::string_view{ file_content_view.data() + pos, file_content_view.size() - pos });
        if (!sv.empty() && !util::starts_with(sv, comment)) {
            lines_.push_back(sv);
        }
    }

    int file_descriptor_ = 0;
    char *file_content_ = nullptr;
    std::size_t size_ = 0;
    std::vector<std::string_view> lines_{};
};

}  // namespace plssvm

#undef PLSSVM_HAS_MEMORY_MAPPING