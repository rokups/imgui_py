%module imgui
%{
#include "imgui.h"
#include "imgui_internal.h"
#include "imgui_impl_sdl.h"
#include "imgui_impl_opengl3.h"
#include "misc/cpp/imgui_stdlib.h"
#include "imgui_extra.h"
#include "DearImGuiApp.h"
#include <vector>
%}

%include "std_string.i"

%rename("%(undercase)s", match$kind="function") "";
%rename("%(undercase)s", match$kind="variable") "";
%rename("%(regex:/^ImGui(.+)$/\\1/)s", %$isenumitem) "";
%rename("%(regex:/^(ImGuiCond|ImGuiCol|ImGuiStyle)(.+)$/\\1\\2/)s", %$isenumitem) "";

%include "imgui_primitives.i"

/// region Common
%apply int* INOUT  {
    int* v
}
%apply bool* INOUT  {
    bool* v,
    bool* p_open,
    bool* p_selected
};
%apply float* INOUT {
    float* v,
    float* v_current_min,
    float* v_current_max,
    float* v_rad
};

%ignore ImGui::TreeNodeV;
%ignore ImGui::TreeNodeExV;
%ignore ImGui::SetTooltipV;
%ignore ImGui::LogTextV;
%ignore ImGuiTextBuffer::appendfv;
/// endregion

/// region Text
%inline
%{
void Text(const char* text) { ImGui::TextUnformatted(text); }
void TextColored(ImColor color, const char* text) { ImGui::TextColored(color, "%s", text); }
void TextDisabled(const char* text) { ImGui::TextDisabled("%s", text); }
void TextWrapped(const char* text) { ImGui::TextWrapped("%s", text); }
void LabelText(const char* text) { ImGui::LabelText("%s", text); }
void BulletText(const char* text) { ImGui::BulletText("%s", text); }
%}
%ignore ImGui::Text;
%ignore ImGui::TextUnformatted;
%ignore ImGui::TextV;
%ignore ImGui::TextColoredV;
%ignore ImGui::TextDisabledV;
%ignore ImGui::TextWrappedV;
%ignore ImGui::LabelTextV;
%ignore ImGui::BulletTextV;
%ignore ImGui::TextColored;
%ignore ImGui::TextDisabled;
%ignore ImGui::TextWrapped;
%ignore ImGui::LabelText;
%ignore ImGui::BulletText;
/// endregion

/// region Menus
%ignore ImGui::MenuItem(char const *,char const *,bool);
%ignore ImGui::MenuItem(char const *,char const *,bool,bool);
/// endregion

/// region Tables
%ignore ImGuiTableSortSpecs::Specs;
%extend ImGuiTableSortSpecs
{
    const ImGuiTableColumnSortSpecs* GetSpec(int i)
    {
        if (i < $self->SpecsCount)
            return &$self->Specs[i];
        return NULL;
    }
}
/// endregion

/// region Combo
%ignore ImGui::Combo;
%apply int* INOUT { int* current_item };
%inline
%{
bool Combo(const char* label, int* current_item, PyObject* items, int popup_max_height_in_items = -1)
{
    struct Context
    {
        PyObject* Items;
        std::vector<PyObject*> Deref;
    } ctx { items };
    if (!PySequence_Check(items))
    {
        PyErr_SetString(PyExc_ValueError, "expected a sequence");
        return false;
    }
    auto getter = [](void* data, int idx, const char** out_text)
    {
        Context* ctx = (Context*)data;
        PyObject* item = PySequence_GetItem(ctx->Items, idx);
        if (PyUnicode_Check(item))
            *out_text = PyUnicode_AsUTF8(item);
        else
        {
            PyObject* str = PyObject_Str(item);
            ctx->Deref.push_back(str);
            *out_text = PyUnicode_AsUTF8(str);
        }
        Py_DECREF(item);
        return *out_text != nullptr;
    };
    bool result = ImGui::Combo(label, current_item, getter, (void*)&ctx, PySequence_Length(items), popup_max_height_in_items);
    for (PyObject* o : ctx.Deref)
        Py_DECREF(o);
    return result;
}
%}
/// endregion

/// region Selectable
// Using a pointer instead of default "size" value allows python wrapper to use actual arg names with
// default values, improving autocompletion.
%ignore ImGui::Selectable;
%inline
%{
bool Selectable(const char* label, bool selected, ImGuiSelectableFlags flags = 0, const ImVec2* size = nullptr)
{
    return ImGui::Selectable(label, selected, flags, size ? *size : ImVec2(0, 0));
}
%}
/// endregion

/// region InputText
%ignore ImGui::InputText;
%ignore ImGui::InputTextMultiline;
%ignore ImGui::InputTextWithHint;
%apply std::string& INOUT { std::string& buf }
%inline
%{
bool InputText(const char* label, std::string& buf, ImGuiInputTextFlags flags = 0)
{
	return ImGui::InputText(label, &buf, flags);
}
//bool InputTextMultiline(const char* label, char* buf, size_t buf_size, const ImVec2& size = ImVec2(0, 0), ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL);
//bool InputTextWithHint(const char* label, const char* hint, char* buf, size_t buf_size, ImGuiInputTextFlags flags = 0, ImGuiInputTextCallback callback = NULL, void* user_data = NULL);
%}
/// endregion

/// region ColorEdit3/ColorPicker3
// Color pickers. We ignore "3" versions and we drop suffix for "4" variants as we always specify full color. There is a flag for not having alpha component.
%ignore ImGui::ColorEdit3;
%ignore ImGui::ColorPicker3;
%rename(ColorEdit) ImGui::ColorEdit3;
%rename(ColorPicker) ImGui::ColorPicker3;
%typemap(in) float col[4](float temp[4]) {
    if (PySequence_Check($input))
    {
        swig::SwigVar_PyObject tuple(PySequence_Tuple($input));
        if (!PyArg_ParseTuple(tuple, "ffff", temp, temp+1, temp+2, temp+3))
        {
            PyErr_SetString(PyExc_TypeError, "tuple must have 4 elements");
            SWIG_fail;
        }
        $1 = &temp[0];
    }
    else
    {
        PyErr_SetString(PyExc_TypeError, "expected a tuple.");
        SWIG_fail;
    }
}
%typemap(argout, noblock=1) float col[4] {
    %append_output(
        PyObject_CallObject(
            swig::SwigVar_PyObject(PyObject_GetAttrString(swig::SwigVar_PyObject(PyImport_ImportModule("imgui")), "ImColor")),
            swig::SwigVar_PyObject(Py_BuildValue("(ffff)", $1[0], $1[1], $1[2], $1[3]))
        )
    );
}
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) float col[4] "$1 = PySequence_Check($input) && PySequence_Length($input) == 4;";
/// endregion

%include "../imconfig.h"
%include "imgui.h"
%include "imgui_impl_sdl.h"
%include "imgui_impl_opengl3.h"
%include "imgui_extra.h"
%include "DearImGuiApp.h"


//%rename("$ignore", regexmatch$name="^IM_.+$") "";
//%rename("$ignore", regexmatch$name="^Im.+$") "";
//%rename("$ignore", regexmatch$name="^ImGui::.+$") "";
//%rename(ImGuiContext) ImGuiContext;
//%ignore ImGuiContext::DockContext;
//%ignore ImGuiContext::InputEventsQueue;
//%ignore ImGuiContext::InputEventsTrail;
//%ignore ImGuiContext::DrawListSharedData;
//%ignore ImGuiContext::Windows;
//%ignore ImGuiContext::WindowsFocusOrder;
//%ignore ImGuiContext::WindowsTempSortBuffer;
//%ignore ImGuiContext::CurrentWindowStack;
//%ignore ImGuiContext::WindowsById;
//%ignore ImGuiContext::KeysOwnerData;
//%ignore ImGuiContext::KeysRoutingTable;
//%ignore ImGuiContext::ColorStack;
//%ignore ImGuiContext::StyleVarStack;
//%ignore ImGuiContext::FontStack;
//%ignore ImGuiContext::FocusScopeStack;
//%ignore ImGuiContext::ItemFlagsStack;
//%ignore ImGuiContext::GroupStack;
//%ignore ImGuiContext::OpenPopupStack;
//%ignore ImGuiContext::BeginPopupStack;
//%ignore ImGuiContext::Viewports;
//%ignore ImGuiContext::DragDropPayloadBufHeap;
//%ignore ImGuiContext::DragDropPayloadBufLocal;
//%ignore ImGuiContext::TablesTempData;
//%ignore ImGuiContext::Tables;
//%ignore ImGuiContext::TablesLastTimeActive;
//%ignore ImGuiContext::DrawChannelsTempMergeBuffer;
//%ignore ImGuiContext::TabBars;
//%ignore ImGuiContext::CurrentTabBarStack;
//%ignore ImGuiContext::ShrinkWidthBuffer;
//%ignore ImGuiContext::ClipboardHandlerData;
//%ignore ImGuiContext::MenusIdSubmittedThisFrame;
//%ignore ImGuiContext::SettingsHandlers;
//%ignore ImGuiContext::SettingsWindows;
//%ignore ImGuiContext::SettingsTables;
//%ignore ImGuiContext::Hooks;
//%ignore ImGuiContext::LocalizationTable;
//%ignore ImGuiContext::TempBuffer;
//%include "ignore_imgui_internal.i"
//%include <imgui_internal.h>

%pythonbegin %{

_ipython_callbacks = []

def ipython_gui(title):
    def decorator(fn):
        global _ipython_callbacks
        _ipython_callbacks.append(lambda: fn(title=title))
    return decorator


def ipython_window(title):
    def decorator(fn):
        global _ipython_callbacks
        def render_window():
            flags = WindowFlags_NoDecoration | WindowFlags_NoResize | WindowFlags_NoMove
            vp = get_main_viewport()
            set_next_window_pos(vp.pos)
            set_next_window_size(vp.size)
            if begin(title, flags=flags):
                do_continue = fn()
                end()
            else:
                do_continue = False
            return do_continue
        _ipython_callbacks.append((title, render_window))
    return decorator

%}