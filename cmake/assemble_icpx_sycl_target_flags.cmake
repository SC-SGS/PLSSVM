# Authors: Alexander Van Craen, Marcel Breyer
# Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
# License: This file is part of the PLSSVM project which is released under the MIT license.
#          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

# function to check whether a string starts with a substring
function (startswith out_var string prefix)
    string(FIND "${string}" "${prefix}" pos)
    if (pos EQUAL 0)
        set(${out_var} ON PARENT_SCOPE)
    else ()
        set(${out_var} OFF PARENT_SCOPE)
    endif ()
endfunction ()

# function to assemble the icpx compiler flags and add them to the target
function (assemble_icpx_sycl_target_flags target scope)
    set(_fsycl_target_archs "")

    # CPU targets
    if (DEFINED PLSSVM_CPU_TARGET_ARCHS)
        # assemble -fsycl-targets
        list(APPEND _fsycl_target_archs "spir64_x86_64")

        # set target arch explicitly
        if (PLSSVM_NUM_CPU_TARGET_ARCHS EQUAL 1)
            target_link_options(${target} ${scope} -Xsycl-target-backend=spir64_x86_64 "-march=${PLSSVM_CPU_TARGET_ARCHS}")
        endif ()
    endif ()

    # NVIDIA GPU targets
    if (DEFINED PLSSVM_NVIDIA_TARGET_ARCHS)
        # assemble -fsycl-targets
        foreach (_arch ${PLSSVM_NVIDIA_TARGET_ARCHS})
            list(APPEND _fsycl_target_archs "nvidia_gpu_${_arch}")
        endforeach ()

        # add lineinfo for easier profiling
        target_link_options(${target} ${scope} -Xcuda-ptxas -lineinfo)
        # add verbose kernel compilation information to output if in Debug mode
        target_link_options(${target} ${scope} $<$<CONFIG:Debug>:-Xcuda-ptxas --verbose>)
    endif ()

    # AMD GPU targets
    if (DEFINED PLSSVM_AMD_TARGET_ARCHS)
        # assemble -fsycl-targets
        foreach (_arch ${PLSSVM_AMD_TARGET_ARCHS})
            list(APPEND _fsycl_target_archs "amd_gpu_${_arch}")
        endforeach ()
    endif ()

    # Intel GPU targets
    if (DEFINED PLSSVM_INTEL_TARGET_ARCHS)
        # iterate over all target archs and check how many of them are provided as HEX values
        set(PLSSVM_INTEL_TARGET_ARCH_NUM_HEX 0)
        foreach (_arch ${PLSSVM_INTEL_TARGET_ARCHS})
            # test whether the arch is a hex value
            startswith(PLSSVM_INTEL_TARGET_ARCH_IS_HEX "${_arch}" "0x")
            if (PLSSVM_INTEL_TARGET_ARCH_IS_HEX)
                math(EXPR PLSSVM_INTEL_TARGET_ARCH_NUM_HEX "${PLSSVM_INTEL_TARGET_ARCH_NUM_HEX} + 1")
            endif ()
        endforeach ()

        # either ALL targets must be provided as HEX values or NONE
        if (PLSSVM_INTEL_TARGET_ARCH_NUM_HEX EQUAL 0)
            # no architecture was provided as HEX value -> use new shortcuts
            foreach (_arch ${PLSSVM_INTEL_TARGET_ARCHS})
                list(APPEND _fsycl_target_archs "intel_gpu_${_arch}")
            endforeach ()
        elseif (PLSSVM_INTEL_TARGET_ARCH_NUM_HEX EQUAL PLSSVM_NUM_INTEL_TARGET_ARCHS)
            if (PLSSVM_NUM_INTEL_TARGET_ARCHS GREATER 1)
                message(
                    FATAL_ERROR
                        "When specifying the Intel architectures with HEX values, only a single architecture is supported but ${PLSSVM_NUM_INTEL_TARGET_ARCHS} where provided!"
                )
            endif ()
            # use old way to specify architectures
            list(APPEND _fsycl_target_archs "spir64_gen")
            list(JOIN PLSSVM_INTEL_TARGET_ARCHS "," PLSSVM_INTEL_TARGET_ARCHS_STRING)
            target_compile_options(${target} ${scope} -Xsycl-target-backend=spir64_gen "-device ${PLSSVM_INTEL_TARGET_ARCHS_STRING}")
            target_link_options(${target} ${scope} -Xsycl-target-backend=spir64_gen "-device ${PLSSVM_INTEL_TARGET_ARCHS_STRING}")
        else ()
            message(
                FATAL_ERROR
                    "The provided Intel GPU target architectures (${PLSSVM_INTEL_TARGET_ARCHS}) are a mixture between device IDs (hex values) and architecture names but only either of them are supported!"
            )
        endif ()
    endif ()

    # apply -fsycl-targets
    list(JOIN _fsycl_target_archs "," _fsycl_target_archs_string)
    if (NOT _fsycl_target_archs_string STREQUAL "")
        message(STATUS "Compiling for -fsycl-targets=${_fsycl_target_archs}")
        target_compile_options(${target} ${scope} -fsycl-targets=${_fsycl_target_archs_string})
        target_link_options(${target} ${scope} -fsycl-targets=${_fsycl_target_archs_string})
    endif ()
endfunction ()
