add_library(implot SHARED
    implot/implot.h
    implot/implot.cpp
    implot/implot_items.cpp
    implot/implot_demo.cpp)
target_link_libraries(implot PUBLIC imgui)
target_include_directories(implot PUBLIC ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/implot)
target_compile_definitions(implot PRIVATE -DIMPLOT_BUILD=1)

set_property(SOURCE implot.i PROPERTY CPLUSPLUS ON)
set_property(SOURCE implot.i PROPERTY INCLUDE_DIRECTORIES ${imgui_py_SOURCE_DIR}/imgui_py/imgui ${imgui_py_SOURCE_DIR}/imgui_py/imgui/backends ${CMAKE_CURRENT_SOURCE_DIR}/implot)
set_property(SOURCE implot.i PROPERTY COMPILE_DEFINITIONS -DIMGUI_USER_CONFIG="${imgui_py_SOURCE_DIR}/imconfig.h")
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
#    -debug-top <n>  # Display entire parse tree at stages 1-4, <n> is a csv list of stages
#    -debug-typedef  # Display information about the types and typedefs in the interface
#    -debug-typemap  # Display typemap debugging information
#    -debug-tmsearch # Display typemap search debugging information
#    -debug-tmused   # Display typemaps used debugging information
#    2>&1 > swig.log
)
swig_add_library(implot_py
    TYPE SHARED
    LANGUAGE Python
	OUTPUT_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    OUTFILE_DIR ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}
    SOURCES implot.i
)
target_link_libraries(implot_py PUBLIC implot Python3::Python Python3::NumPy)
target_compile_definitions(implot_py PUBLIC NPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION)

if (IMGUI_PY_INSTALL)
    install(TARGETS implot_py LIBRARY DESTINATION .)
    install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/implot.py DESTINATION bin)
    install(FILES ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/$<CONFIG>/implot.dll DESTINATION bin)
endif ()
