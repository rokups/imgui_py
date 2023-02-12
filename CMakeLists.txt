cmake_minimum_required(VERSION 3.19)
cmake_policy(SET CMP0078 NEW)
cmake_policy(SET CMP0086 NEW)
project(imgui_py)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

add_subdirectory(imgui_py)
add_subdirectory(implot_py)
