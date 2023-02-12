#pragma once

// TODO: IMGUI_API for windows

#define IMGUI_DISABLE_OBSOLETE_FUNCTIONS
#define IMGUI_DISABLE_OBSOLETE_KEYIO
#define IMPLOT_DISABLE_OBSOLETE_FUNCTIONS
#define ImDrawIdx unsigned int
//#define IMGUI_USE_WCHAR32
#define IM_VEC4_CLASS_EXTRA \
    float operator[] (size_t idx) const { return (&x)[idx]; } \
    float& operator[] (size_t idx) { return (&x)[idx]; }
