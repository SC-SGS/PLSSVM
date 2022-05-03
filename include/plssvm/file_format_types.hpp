#pragma once

#include <iosfwd>  // forward declare std::ostream and std::istream

namespace plssvm {

enum class file_format_type {
    libsvm,
    arff
};

/**
 * @brief Output the @p format to the given output-stream @p out.
 * @param[in,out] out the output-stream to write the format type to
 * @param[in] format the file format type
 * @return the output-stream
 */
std::ostream &operator<<(std::ostream &out, file_format_type format);

/**
 * @brief Use the input-stream @p in to initialize the @p format type.
 * @param[in,out] in input-stream to extract the format type from
 * @param[in] format the file format type
 * @return the input-stream
 */
std::istream &operator>>(std::istream &in, file_format_type &format);

}