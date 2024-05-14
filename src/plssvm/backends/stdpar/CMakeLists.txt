## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

list(APPEND CMAKE_MESSAGE_INDENT "stdpar:  ")

# check if stdpar can be enabled
message(CHECK_START "Checking for stdpar backend")

# explicitly set general sources
set(PLSSVM_STDPAR_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.cpp
    ${CMAKE_CURRENT_LIST_DIR}/csvm.cpp
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.cpp
    ${CMAKE_CURRENT_LIST_DIR}/implementation_types.cpp
)

# create interface library used in the sub-CMakeLists
set(PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE plssvm-stdpar-interface)
add_library(${PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE} INTERFACE)

set(PLSSVM_STDPAR_BACKEND_IMPLEMENTATION AUTO CACHE STRING "Set the stdpar implementation that should be used.")
set_property(CACHE PLSSVM_STDPAR_BACKEND_IMPLEMENTATION PROPERTY STRINGS AUTO NVHPC roc-stdpar IntelLLVM ACPP GNU_TBB)

# try finding a stdpar backend
if (PLSSVM_STDPAR_BACKEND_IMPLEMENTATION MATCHES "NVHPC")
    # NVHPC requested -> try finding it
    add_subdirectory(NVHPC)
    # check whether it could be found
    if (NOT PLSSVM_STDPAR_BACKEND)
        message(SEND_ERROR "Cannot find requested backend: stdpar with NVHPC!")
        return()
    endif ()
elseif (PLSSVM_STDPAR_BACKEND_IMPLEMENTATION MATCHES "roc-stdpar")
    # roc-stdpar requested -> try finding it
    add_subdirectory(roc-stdpar)
    # check whether it could be found
    if (NOT PLSSVM_STDPAR_BACKEND)
        message(SEND_ERROR "Cannot find requested backend: stdpar with roc-stdpar!")
        return()
    endif ()
elseif (PLSSVM_STDPAR_BACKEND_IMPLEMENTATION MATCHES "IntelLLVM")
    # Intel LLVM requested -> try finding it
    add_subdirectory(IntelLLVM)
    # check whether it could be found
    if (NOT PLSSVM_STDPAR_BACKEND)
        message(SEND_ERROR "Cannot find requested backend: stdpar with IntelLLVM!")
        return()
    endif ()
elseif (PLSSVM_STDPAR_BACKEND_IMPLEMENTATION MATCHES "ACPP")
    # AdaptiveCpp requested -> try finding it
    add_subdirectory(AdaptiveCpp)
    # check whether it could be found
    if (NOT PLSSVM_STDPAR_BACKEND)
        message(SEND_ERROR "Cannot find requested backend: stdpar with ACPP!")
        return()
    endif ()
elseif (PLSSVM_STDPAR_BACKEND_IMPLEMENTATION MATCHES "GNU_TBB")
    # GNU GCC + TBB requested -> try finding it
    add_subdirectory(GNU_TBB)
    # check whether it could be found
    if (NOT PLSSVM_STDPAR_BACKEND)
        message(SEND_ERROR "Cannot find requested backend: stdpar with GNU_TBB!")
        return()
    endif ()
elseif (PLSSVM_STDPAR_BACKEND_IMPLEMENTATION MATCHES "AUTO")
    # automatic order is NVHPC -> Intel LLVM -> AdaptiveCpp -> GNU GCC + TBB
    add_subdirectory(NVHPC)
    if (NOT PLSSVM_STDPAR_BACKEND)
        # NVHPC not found -> try finding roc-stdpar
        add_subdirectory(roc-stdpar)
        if (NOT PLSSVM_STDPAR_BACKEND)
            # roc-stdpar not found -> try finding Intel LLVM
            add_subdirectory(IntelLLVM)
            if (NOT PLSSVM_STDPAR_BACKEND)
                # Intel LLVM not found -> try finding AdaptiveCpp
                add_subdirectory(AdaptiveCpp)
                if (NOT PLSSVM_STDPAR_BACKEND)
                    # AdaptiveCpp not found -> try finding GNU GCC + TBB
                    add_subdirectory(GNU_TBB)
                    if (NOT PLSSVM_STDPAR_BACKEND)
                        # no stdpar backend found
                        message(CHECK_FAIL "not found")
                        return()
                    endif ()
                endif ()
            endif ()
        endif ()
    endif ()
else ()
    message(CHECK_FAIL "not found")
    return()
endif ()

# create real stdpar library
set_local_and_parent(PLSSVM_STDPAR_BACKEND_LIBRARY_NAME plssvm-stdpar)
add_library(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} STATIC ${PLSSVM_STDPAR_SOURCES})

# link against interface library
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE})

# set potential compiler flags in parent scope
if (PLSSVM_STDPAR_BACKEND_COMPILER_FLAG)
    message(STATUS "Adding stdpar ${PLSSVM_STDPAR_BACKEND} specific compiler flag(s) \"${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG}\" to CMAKE_CXX_FLAGS.")
    set_local_and_parent(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${PLSSVM_STDPAR_BACKEND_COMPILER_FLAG}")
endif ()
set_local_and_parent(PLSSVM_STDPAR_BACKEND ${PLSSVM_STDPAR_BACKEND})

# link base library against stdpar library
target_link_libraries(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_LIBRARY_NAME})

# set compile definition that the stdpar backend is available
target_compile_definitions(${PLSSVM_BASE_LIBRARY_NAME} PUBLIC PLSSVM_HAS_STDPAR_BACKEND)
target_compile_definitions(${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME} PUBLIC PLSSVM_HAS_STDPAR_BACKEND)

# link against interface library
target_link_libraries(${PLSSVM_ALL_LIBRARY_NAME} INTERFACE ${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME})

# mark backend library as install target
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_STDPAR_BACKEND_LIBRARY_NAME})
append_local_and_parent(PLSSVM_TARGETS_TO_INSTALL ${PLSSVM_STDPAR_BACKEND_LIBRARY_INTERFACE})

# generate summary string
include(${PROJECT_SOURCE_DIR}/cmake/assemble_summary_string.cmake)
assemble_summary_string(PLSSVM_STDPAR_BACKEND_SUMMARY_STRING_ARCHS)
if (PLSSVM_STDPAR_BACKEND STREQUAL "NVHPC")
    set(PLSSVM_STDPAR_BACKEND_NAME "nvhpc (nvc++)")
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "roc-stdpar")
    set(PLSSVM_STDPAR_BACKEND_NAME "roc-stdpar")
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "IntelLLVM")
    set(PLSSVM_STDPAR_BACKEND_NAME "Intel LLVM (icpx)")
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "ACPP")
    set(PLSSVM_STDPAR_BACKEND_NAME "AdaptiveCpp")
elseif (PLSSVM_STDPAR_BACKEND STREQUAL "GNU_TBB")
    set(PLSSVM_STDPAR_BACKEND_NAME "GNU GCC + TBB")
endif ()
set(PLSSVM_STDPAR_BACKEND_SUMMARY_STRING " - stdpar (${PLSSVM_STDPAR_BACKEND_NAME}):${PLSSVM_STDPAR_BACKEND_SUMMARY_STRING_ARCHS}" PARENT_SCOPE)

list(POP_BACK CMAKE_MESSAGE_INDENT)