find_package(SDL2 REQUIRED)
find_package(OpenGL REQUIRED)

add_library(imgui SHARED
    imgui/imgui.cpp
    imgui/imgui_demo.cpp
    imgui/imgui_draw.cpp
    imgui/imgui_tables.cpp
    imgui/imgui_widgets.cpp
    imgui/backends/imgui_impl_sdl.cpp
    imgui/backends/imgui_impl_opengl3.cpp
    imgui/misc/cpp/imgui_stdlib.cpp)
target_include_directories(imgui PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui
    ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends)
    target_link_libraries(imgui PUBLIC Python3::Python OpenGL::GL SDL2::SDL2)
target_compile_definitions(imgui PUBLIC -DIMGUI_USER_CONFIG="${imgui_py_SOURCE_DIR}/imconfig.h" PRIVATE -DIMGUI_BUILD=1)

set_property(SOURCE imgui.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE imgui.i PROPERTY INCLUDE_DIRECTORIES ${CMAKE_CURRENT_SOURCE_DIR}/imgui ${CMAKE_CURRENT_SOURCE_DIR}/imgui/backends)
set_property(SOURCE imgui.i PROPERTY COMPILE_DEFINITIONS -DIMGUI_USER_CONFIG="${imgui_py_SOURCE_DIR}/imconfig.h")
set(CMAKE_SWIG_FLAGS
#    [Debug]
#    -debug-classes  # Display information about the classes found in the interface
#    -debug-module <n># Display module parse tree at stages 1-4, <n> is a csv list of stages
#    -debug-symtabs  # Display symbol tables information
#    -debug-symbols  # Display target language symbols in the symbol tables
#    -debug-csymbols # Display C symbols in the symbol tables
#    -debug-lsymbols # Display target language layer symbols
#    -debug-tags     # Display information about the tags found in the interface
#    -debug-template # Display information for debugging templates
#    -debug-top 1  # Display entire parse tree at stages 1-4, <n> is a csv list of stages
#    -debug-typedef  # Display information about the types and typedefs in the interface
#    -debug-typemap  # Display typemap debugging information
#    -debug-tmsearch # Display typemap search debugging information
#    -debug-tmused   # Display typemaps used debugging information
#    2>&1 > swig.log
)
swig_add_library(imgui_py
    TYPE SHARED
    LANGUAGE Python
	OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    OUTFILE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    SOURCES
        imgui.i
        imgui_extra.cpp
        DearImGuiApp.cpp
        DearImGuiApp.h)
target_link_libraries(imgui_py PUBLIC Python3::Python imgui)
target_include_directories(imgui_py PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if (IMGUI_PY_INSTALL)
    install(TARGETS imgui_py LIBRARY DESTINATION .)
    install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/imgui.py DESTINATION bin)
    if (MSVC)
        install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/SDL2.dll DESTINATION bin)
        install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/imgui.dll DESTINATION bin)
    endif ()
endif ()
