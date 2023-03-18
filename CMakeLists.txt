cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
project(imgui_py)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

if (SWIG_EXECUTABLE)
	get_filename_component(SWIG_EXECUTABLE "${SWIG_EXECUTABLE}" ABSOLUTE)
	execute_process(COMMAND "${SWIG_EXECUTABLE}" -swiglib OUTPUT_VARIABLE SWIG_DIR)
	string(STRIP "${SWIG_DIR}" SWIG_DIR)
	# Propagate these variables to global scope
	set(SWIG_EXECUTABLE "${SWIG_EXECUTABLE}" CACHE STRING "" FORCE)
	set(SWIG_DIR "${SWIG_DIR}" CACHE STRING "" FORCE)
endif ()

if (NOT DEFINED IMGUI_PY_INSTALL)
	set (IMGUI_PY_INSTALL ON)
endif ()

if (ENV{VIRTUAL_ENV})
    set(Python3_FIND_VIRTUALENV ONLY)
endif ()

find_package(Python3 REQUIRED COMPONENTS Development NumPy)
find_package(SWIG REQUIRED)
include(UseSWIG)

add_subdirectory(imgui_py)
add_subdirectory(implot_py)
