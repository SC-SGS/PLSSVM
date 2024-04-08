## Authors: Alexander Van Craen, Marcel Breyer
## Copyright (C): 2018-today The PLSSVM project - All Rights Reserved
## License: This file is part of the PLSSVM project which is released under the MIT license.
##          See the LICENSE.md file in the project root for full license information.
########################################################################################################################

## create HIP tests
set(PLSSVM_HIP_TEST_NAME HIP_tests)

include(CheckLanguage)
check_language(HIP)
enable_language(HIP)
find_package(HIP REQUIRED)

# list all necessary sources
set(PLSSVM_HIP_TEST_SOURCES
    ${CMAKE_CURRENT_LIST_DIR}/detail/device_ptr.hip
    ${CMAKE_CURRENT_LIST_DIR}/detail/pinned_memory.hip
    ${CMAKE_CURRENT_LIST_DIR}/kernel/detail/fill_kernel.hip
    ${CMAKE_CURRENT_LIST_DIR}/detail/utility.hip
    ${CMAKE_CURRENT_LIST_DIR}/exceptions.hip
    ${CMAKE_CURRENT_LIST_DIR}/hip_csvm.hip
)

# set target properties
set_local_and_parent(PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME plssvm-HIP_test)
add_library(${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME} SHARED ${PLSSVM_HIP_TEST_SOURCES})

# set target architecture
set_property(TARGET ${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME} PROPERTY HIP_ARCHITECTURES ${PLSSVM_AMD_TARGET_ARCHS})

# link base library against HIP library
target_link_libraries(${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME} PRIVATE hip::device hip::host)
set_target_properties(${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME} PROPERTIES LINKER_LANGUAGE HIP)

target_link_libraries(${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME} PUBLIC ${PLSSVM_BASE_TEST_LIBRARY_NAME})

# add test executable
add_executable(${PLSSVM_HIP_TEST_NAME} ${CMAKE_CURRENT_LIST_DIR}/../../main.cpp)
target_link_libraries(${PLSSVM_HIP_TEST_NAME} PRIVATE ${PLSSVM_HIP_BACKEND_TEST_LIBRARY_NAME})


# add tests to google test
include(GoogleTest)
include(${PROJECT_SOURCE_DIR}/cmake/discover_tests_with_death_test_filter.cmake)
discover_tests_with_death_test_filter(${PLSSVM_HIP_TEST_NAME})

# add test as coverage dependency
if (TARGET coverage)
    add_dependencies(coverage ${PLSSVM_HIP_TEST_NAME})
endif ()