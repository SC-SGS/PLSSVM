function(assemble_summary_string out_var)
    set(PLSSVM_SUMMARY_STRING_ASSEMBLE "")
    if(DEFINED PLSSVM_CPU_TARGET_ARCHS)
        # add cpu platform
        string(APPEND PLSSVM_SUMMARY_STRING_ASSEMBLE " cpu,")
    endif()
    if(DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        # add nvidia platform
        string(APPEND PLSSVM_SUMMARY_STRING_ASSEMBLE " nvidia (${PLSSVM_NVIDIA_TARGET_ARCHS}),")
    endif()
    if(DEFINED PLSSVM_AMD_TARGET_ARCHS)
        # add amd platform
        string(APPEND PLSSVM_SUMMARY_STRING_ASSEMBLE " amd (${PLSSVM_AMD_TARGET_ARCHS}),")
    endif()
    if(DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        # add intel platform
        string(APPEND PLSSVM_SUMMARY_STRING_ASSEMBLE " intel,")
    endif()
    # remove last comma
    string(REGEX REPLACE ",$" "" PLSSVM_SUMMARY_STRING_ASSEMBLE "${PLSSVM_SUMMARY_STRING_ASSEMBLE}")
    set(${out_var} "${PLSSVM_SUMMARY_STRING_ASSEMBLE}" PARENT_SCOPE)
endfunction()