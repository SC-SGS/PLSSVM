function(auto_detect_cuda_arch OUT_LIST)
    # get GPU architecture by querying device information
    try_run(run_result compile_result ${PROJECT_BINARY_DIR} ${PROJECT_SOURCE_DIR}/cmake/compile_tests/compute_capability.cu
        RUN_OUTPUT_VARIABLE compute_capabilities)

    # strip unnecessary information
    string(REGEX MATCHALL "[0-9]+\\.[0-9]+" compute_capabilities "${compute_capabilities}")

    # reformat string to be usable in CUDA_ARCHITECTURES
    string(STRIP "${compute_capabilities}" compute_capabilities_stripped)
    string(REPLACE "." "" CUDA_ARCH_LIST "${compute_capabilities_stripped}")
    list(TRANSFORM CUDA_ARCH_LIST PREPEND "sm_")

    if(CUDA_ARCH_LIST MATCHES ".*PTX.*")
        # couldn't find any CUDA capable GPU
        message(FATAL_ERROR "No CUDA capable GPU was found! Please explicitly specify CUDA architectures to compile for.")
    else()
        # "return" list of found architectures
        set(${OUT_LIST} "${CUDA_ARCH_LIST}" PARENT_SCOPE)
    endif()
endfunction()
