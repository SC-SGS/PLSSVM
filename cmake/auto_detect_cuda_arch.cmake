## see: https://stackoverflow.com/questions/68223398/how-can-i-get-cmake-to-automatically-detect-the-value-for-cuda-architectures ##

## ATTENTION: uses a non public API file -> may silently break in newer CMake versions (but relatively unlikely)

function(auto_detect_cuda_arch OUT_LIST)
    # include select_compute_arch
    include(FindCUDA/select_compute_arch)
    # auto detect installed GPUs
    CUDA_DETECT_INSTALLED_GPUS(INSTALLED_GPU_CCS_1)
    # reformat string to be usable in CUDA_ARCHITECTURES
    string(STRIP "${INSTALLED_GPU_CCS_1}" INSTALLED_GPU_CCS_2)
    string(REPLACE "." "" CUDA_ARCH_LIST "${INSTALLED_GPU_CCS_2}")

    # couldn't find any CUDA capable GPU
    if(CUDA_ARCH_LIST MATCHES ".*PTX.*")
        message(STATUS "No CUDA capable GPU was found!")
        return()
    endif()

    # "return" list of found architectures
    set(${OUT_LIST} "${CUDA_ARCH_LIST}" PARENT_SCOPE)
endfunction()