#pragma once

#ifdef _WIN32
#ifdef IMGUI_BUILD
#define IMGUI_API __declspec(dllexport)
#else
#define IMGUI_API __declspec(dllimport)
#endif	// IMGUI_BUILD

#ifdef IMPLOT_BUILD
#define IMPLOT_API __declspec(dllexport)
#else
#define IMPLOT_API __declspec(dllimport)
#endif	// IMGUI_BUILD
#endif	// _WIN32

#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS
#define IMGUI_DISABLE_OBSOLETE_KEYIO
#define IMPLOT_DISABLE_OBSOLETE_FUNCTIONS
#define ImDrawIdx unsigned int
//#define IMGUI_USE_WCHAR32
#define IM_VEC4_CLASS_EXTRA \
    float operator[] (size_t idx) const { return (&x)[idx]; } \
    float& operator[] (size_t idx) { return (&x)[idx]; }
