#pragma once
#include <iostream>
#include <vector>

#include <plssvm/typedef.hpp>

#include <algorithm>
#include <functional>
#include <numeric>

// TODO: check all, check matching sizes?

template <typename T>
[[nodiscard]] inline std::vector<T> operator-(const std::vector<T> &vec1, const std::vector<T> &vec2) {
    std::vector<T> result(vec1.size());
    for (std::size_t i = 0; i < vec1.size(); ++i) {
        result[i] = vec1[i] - vec2[i];
    }
    return result;
}

template <typename T>
[[nodiscard]] inline std::vector<T> operator+(const std::vector<T> &vec1, const std::vector<T> &vec2) {
    std::vector<T> result(vec1.size());
    for (std::size_t i = 0; i < vec1.size(); ++i) {
        result[i] = (vec1[i] + vec2[i]);
    }
    return result;
}

template <typename T>
inline std::vector<T> &operator+=(std::vector<T> &result, const std::vector<T> &vec2) {
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] += vec2[i];
    }
    return result;
}

template <typename T>
inline T *operator+=(T *result, const std::vector<T> &vec2) {
    for (std::size_t i = 0; i < vec2.size(); ++i) {
        result[i] += vec2[i];
    }
    return result;
}

template <typename T>
[[nodiscard]] inline std::vector<T> operator*(const std::vector<std::vector<T>> &matr, const std::vector<T> &vec) {
    std::vector<T> result(matr.size(), 0.0);
    for (std::size_t i = 0; i < matr.size(); ++i) {
        for (std::size_t j = 0; j < vec.size(); ++j) {
            result[i] += matr[i][j] * vec[j];
        }
    }
    return result;
}

template <typename T>
[[nodiscard]] inline T operator*(const std::vector<T> &vec1, const std::vector<T> &vec2) {
    T result = 0.0;
    for (std::size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

template <typename T>
[[nodiscard]] inline T operator*(T *vec1, const std::vector<T> &vec2) {
    T result = 0.0;
    for (std::size_t i = 0; i < vec2.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

//inline std::ostream &operator<<(std::ostream &out, const std::vector<real_t> &vec) {
//    char buffer[20];
//    for (unsigned i = 0; i < vec.size(); ++i) {
//        if (vec[i] != 0) {
//            sprintf(buffer, "%i:%e ", i, vec[i]);
//            out << buffer;
//        }  //out << i << ":" << vec[i] << " ";
//    }
//    return out;
//}
//
//inline std::ostream &operator<<(std::ostream &out, const std::vector<std::vector<real_t>> &matr) {
//    for (unsigned i = 0; i < matr.size(); ++i) {
//        out << matr[i] << '\n';
//    }
//    return out;
//}

template <typename T>
[[nodiscard]] inline std::vector<T> operator*(const T &value, std::vector<T> vec) {
    for (std::size_t i = 0; i < vec.size(); ++i) {
        vec[i] *= value;
    }
    return vec;
}

template <typename T>
[[nodiscard]] inline T operator*(const std::vector<T> &vec1, T *vec2) {
    T result = 0.0;
    for (std::size_t i = 0; i < vec1.size(); ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

template <typename T>
[[nodiscard]] inline std::vector<T> operator*(const std::vector<T> &vec, const T &value) {
    return value * vec;
}

template <typename T>
[[nodiscard]] inline std::vector<std::vector<T>> dot(const std::vector<T> &vec1, const std::vector<T> &vec2) {
    std::vector<std::vector<T>> result(vec1.size(), std::vector<T>(vec1.size(), 0));
    for (std::size_t i = 0; i < vec1.size(); ++i) {
        for (std::size_t j = 0; j < vec1.size(); ++j) {
            result[i][j] += vec1[i] * vec2[j];
        }
    }
    return result;
}

template <typename T>
inline std::vector<std::vector<T>> &operator-=(std::vector<std::vector<T>> &result, const std::vector<std::vector<T>> &matr) {
    for (std::size_t i = 0; i < result.size(); ++i) {
        for (std::size_t j = 0; j < result[0].size(); ++j) {
            result[i][j] -= matr[i][j];
        }
    }
    return result;
}

template <typename T>
inline std::vector<std::vector<T>> &operator+=(std::vector<std::vector<T>> &result, const T &value) {
    for (std::size_t i = 0; i < result.size(); ++i) {
        for (std::size_t j = 0; j < result[0].size(); ++j) {
            result[i][j] += value;
        }
    }
    return result;
}

template <typename T>
inline std::vector<T> &operator-=(std::vector<T> &result, const T &value) {
    for (std::size_t i = 0; i < result.size(); ++i) {
        result[i] -= value;
    }
    return result;
}

template <typename T>
[[nodiscard]] inline T sum(std::vector<T> &vec) {
    T result = 0;
    for (std::size_t i = 0; i < vec.size(); ++i) {
        result += vec[i];
    }
    return result;
}

template <typename T>
[[nodiscard]] inline T mult(T *vec1, T *vec2, std::size_t dim) {
    std::remove_const_t<T> result = 0.0;
    for (std::size_t i = 0; i < dim; ++i) {
        result += vec1[i] * vec2[i];
    }
    return result;
}

template <typename T>
[[nodiscard]] inline T *mult(T value, T *vec, std::size_t dim) {
    for (std::size_t i = 0; i < dim; ++i) {
        vec[i] *= value;
    }
    return vec;
}

template <typename T>
[[nodiscard]] inline T *mult(T *vec, T val, std::size_t dim) {
    return mult(val, vec, dim);
}

template <typename T>
[[nodiscard]] inline T *add(T value, T *vec, std::size_t dim) {
    for (std::size_t i = 0; i < dim; ++i) {
        vec[i] += value;
    }
    return vec;
}

template <typename T>
[[nodiscard]] inline T *add(T *vec1, T *vec2, std::size_t dim) {  // TODO: BBBBBAAAAADDDDDDD
    T *result = new T[dim];
    for (unsigned i = 0; i < dim; ++i) {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}

template <typename T>
inline T *add(T *vec1, T *vec2, T *result, std::size_t dim) {
    for (std::size_t i = 0; i < dim; ++i) {
        result[i] = vec1[i] + vec2[i];
    }
    return result;
}

template <typename T>
[[nodiscard]] inline T *add(T *vec, T value, std::size_t dim) {
    return add(value, vec, dim);
}

template <typename T>
inline std::vector<T> &operator+=(std::vector<T> &vec1, T *vec2) {
    for (std::size_t i = 0; i < vec1.size(); ++i) {
        vec1[i] += vec2[i];
    }
    return vec1;
}