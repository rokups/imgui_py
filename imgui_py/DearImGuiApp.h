#pragma once

#include <string>

struct SDL_Window;

class DearImGuiApp
{
public:
    DearImGuiApp(const char* title, int width = 0, int height = 0, int x = INT32_MAX, int y = INT32_MAX);
    ~DearImGuiApp();
    bool BeginFrame();
    void EndFrame();
    void Close();

    std::string Title;
	int Width;
	int Height;
	int X;
	int Y;

private:
    SDL_Window* Window = nullptr;
    void* GlContext = nullptr;
};
