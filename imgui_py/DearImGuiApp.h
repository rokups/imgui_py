#pragma once

#include <string>

struct SDL_Window;

class DearImGuiApp
{
public:
    DearImGuiApp(const char* title, int width = 0, int height = 0);
    ~DearImGuiApp();
    bool BeginFrame();
    void EndFrame();
    void Close();

    std::string Title;

private:
    SDL_Window* Window = nullptr;
    void* GlContext = nullptr;
};
