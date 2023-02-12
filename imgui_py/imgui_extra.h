#pragma once

#include "imgui.h"
#include <string>

namespace ImGui
{
IMGUI_API bool  InputText(const char* label, ImGuiTextBuffer* str, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL);
IMGUI_API bool  InputTextMultiline(const char* label, ImGuiTextBuffer* str, const ImVec2& size = ImVec2(0, 0), ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL);
IMGUI_API bool  InputTextWithHint(const char* label, const char* hint, ImGuiTextBuffer* str, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL);
}
