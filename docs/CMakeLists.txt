### based on https://vicrucann.github.io/tutorials/quick-cmake-doxygen/ ###

# find doxygen
find_package(Doxygen REQUIRED)

# configure doxygen files
set(DOXYGEN_IN Doxyfile.in)
set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile.out)
configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)


add_custom_command(
  OUTPUT
    doxygen.stamp
  COMMAND
    ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
  COMMENT
    "Generating API documentation with Doxygen"
  VERBATIM
)

add_custom_target(doc DEPENDS doxygen.stamp)