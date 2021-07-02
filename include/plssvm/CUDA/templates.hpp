#include <cassert>
#include <vector>

namespace plssvm {

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s:%d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// CUDA malloc
template <typename T>
inline void cuda_malloc(T data) {
    cuda_malloc(&data, 1);
}

// template <typename T>
// inline void cuda_malloc(std::vector<T>& vec) {
//   cuda_malloc(vec.data(), vec.size());
// }

// template <typename T, typename S, std::enable_if_t<std::is_integral<S>::value, int> = 0>
// inline void cuda_malloc(std::vector<T>& vec, const S size) {
//   cuda_malloc(vec.data(), size);
// }

template <typename T, typename S, std::enable_if_t<std::is_integral<S>::value, int> = 0>
inline void cuda_malloc(T *&ptr, const S size) {
    gpuErrchk(cudaMalloc((void **) &ptr, size * sizeof(T)));
}

// CUDA memset
template <typename T>
inline void cuda_memset(std::vector<T> &vec, const int value) {
    cuda_memset(vec.data(), value, vec.size());
}

template <typename T, typename S, std::enable_if_t<std::is_integral<S>::value, int> = 0>
inline void cuda_memset(std::vector<T> &vec, const int value, const S size) {
    cuda_memset(vec.data(), value, size);
}

template <typename T, typename S, std::enable_if_t<std::is_integral<S>::value, int> = 0>
inline void cuda_memset(T *const &ptr, const int value, const S size) {
    gpuErrchk(cudaMemset(ptr, value, size * sizeof(T)));
}

// CUDA memcpy
template <typename T>
inline void cuda_memcpy(std::vector<T> &vec_dst, std::vector<T> &vec_src, const cudaMemcpyKind kind) {
    assert((vec_dst.size() == vec_src.size()));
    cuda_memcpy(vec_dst.data(), vec_src.data(), vec_src.size(), kind);
}

template <typename T, typename S, std::enable_if_t<std::is_integral<S>::value, int> = 0>
inline void cuda_memcpy(std::vector<T> &vec_dst,
                        const std::vector<T> &vec_src,
                        const S size,
                        const cudaMemcpyKind kind) {
    cuda_memcpy(vec_dst.data(), vec_src.data(), size, kind);
}

template <typename T, typename S, std::enable_if_t<std::is_integral<S>::value, int> = 0>
inline void cuda_memcpy(T *const &ptr_dst, const std::vector<T> &vec_src, const S size, const cudaMemcpyKind kind) {
    cuda_memcpy(ptr_dst, vec_src.data(), size, kind);
}

template <typename T, typename S, std::enable_if_t<std::is_integral<S>::value, int> = 0>
inline void cuda_memcpy(std::vector<T> &vec_dst, const T *const &ptr_src, const S size, const cudaMemcpyKind kind) {
    cuda_memcpy(vec_dst.data(), ptr_src, size, kind);
}

template <typename T>
inline void cuda_memcpy(T &dst,
                        const T *const &ptr_src,
                        const cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyDeviceToHost) {
    assert(kind == cudaMemcpyKind::cudaMemcpyDeviceToHost);
    cuda_memcpy(&dst, ptr_src, 1, kind);
}

template <typename T>
inline void cuda_memcpy(T *const &ptr_dst,
                        const T &src,
                        const cudaMemcpyKind kind = cudaMemcpyKind::cudaMemcpyHostToDevice) {
    assert(kind == cudaMemcpyKind::cudaMemcpyHostToDevice);
    cuda_memcpy(ptr_dst, &src, 1, kind);
}

template <typename T, typename S, std::enable_if_t<std::is_integral<S>::value, int> = 0>
inline void cuda_memcpy(T *const &ptr_dst, const T *const &ptr_src, const S size, const cudaMemcpyKind kind) {
    gpuErrchk(cudaMemcpy(ptr_dst, ptr_src, size * sizeof(T), kind));
}

// CUDA memcpy peer
template <typename T, typename D1, typename D2, std::enable_if_t<std::is_integral<D1>::value, int> = 0, std::enable_if_t<std::is_integral<D2>::value, int> = 0>
inline void cuda_memcpy(std::vector<T> &vec_dst,
                        const D1 device_dest,
                        std::vector<T> &vec_src,
                        const D2 device_src,
                        const cudaMemcpyKind kind) {
    assert((vec_dst.size() == vec_src.size()));
    cuda_memcpy(vec_dst.data(), device_dest, vec_src.data(), device_src, vec_src.size(), kind);
}

template <typename T, typename S, typename D1, typename D2, std::enable_if_t<std::is_integral<S>::value, int> = 0, std::enable_if_t<std::is_integral<D1>::value, int> = 0, std::enable_if_t<std::is_integral<D2>::value, int> = 0>
inline void cuda_memcpy(std::vector<T> &vec_dst,
                        const D1 device_dest,
                        const std::vector<T> &vec_src,
                        const D2 device_src,
                        const S size,
                        const cudaMemcpyKind kind) {
    cuda_memcpy(vec_dst.data(), device_dest, vec_src.data(), device_src, size, kind);
}

template <typename T, typename S, typename D1, typename D2, std::enable_if_t<std::is_integral<S>::value, int> = 0, std::enable_if_t<std::is_integral<D1>::value, int> = 0, std::enable_if_t<std::is_integral<D2>::value, int> = 0>
inline void cuda_memcpy(T *const &ptr_dst,
                        const D1 device_dest,
                        const std::vector<T> &vec_src,
                        const D2 device_src,
                        const S size,
                        const cudaMemcpyKind kind) {
    cuda_memcpy(ptr_dst, device_dest, vec_src.data(), device_src, size, kind);
}

template <typename T, typename S, typename D1, typename D2, std::enable_if_t<std::is_integral<S>::value, int> = 0, std::enable_if_t<std::is_integral<D1>::value, int> = 0, std::enable_if_t<std::is_integral<D2>::value, int> = 0>
inline void cuda_memcpy(std::vector<T> &vec_dst,
                        const D1 device_dest,
                        const T *const &ptr_src,
                        const D2 device_src,
                        const S size,
                        const cudaMemcpyKind kind) {
    cuda_memcpy(vec_dst.data(), device_dest, ptr_src, device_src, size, kind);
}

template <typename T, typename S, typename D1, typename D2, std::enable_if_t<std::is_integral<S>::value, int> = 0, std::enable_if_t<std::is_integral<D1>::value, int> = 0, std::enable_if_t<std::is_integral<D2>::value, int> = 0>
inline void cuda_memcpy(T *const &ptr_dst,
                        const D1 device_dest,
                        const T *const &ptr_src,
                        const D2 device_src,
                        const S size) {
    gpuErrchk(cudaMemcpyPeer(ptr_dst, device_dest, ptr_src, device_src, size * sizeof(T)));
}

// CUDA setDevice
inline void cuda_set_device(int device = 0) {
    gpuErrchk(cudaSetDevice(device));
}

//CUDA Device Synchronize
inline void cuda_device_synchronize() {
    gpuErrchk(cudaDeviceSynchronize());
}

}  // namespace plssvm