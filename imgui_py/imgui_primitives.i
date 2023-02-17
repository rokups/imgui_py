%pythonbegin %{
import sys
import math
from collections import namedtuple
imgui = sys.modules[__name__]
__ImVec2__ = namedtuple('ImVec2', ['x', 'y'])
ImVec4 = namedtuple('ImVec4', ['x', 'y', 'z', 'w'])
__ImColor__ = namedtuple('ImColor', ['r', 'g', 'b', 'a'])
__ImRect__ = namedtuple('ImRect', ['min', 'max'])


class ImColor(__ImColor__):
    @property
    def u32(self):
        out  = (int(self.r * 255.0 + 0.5)) << 0
        out |= (int(self.g * 255.0 + 0.5)) << 8
        out |= (int(self.b * 255.0 + 0.5)) << 16
        out |= (int(self.a * 255.0 + 0.5)) << 24
        return out


class ImVec2(__ImVec2__):
    #def __init__(self, x=0.0, y=0.0):
    #    self.x = x
    #    self.y = y
    #
    #def __iter__(self):
    #    return iter((self.x, self.y))

    @property
    def length(self):
        return math.sqrt((self.x * self.x) + (self.y * self.y))

    def normalize(self):
        l = self.length
        if l > 0:
            return ImVec2(self.x / l, self.y / l)
        return ImVec2(0, 0)

    def __add__(self, rhs):
        if isinstance(rhs, (float, int)):
            return ImVec2(self.x + rhs, self.y + rhs)
        return ImVec2(self.x + rhs.x, self.y + rhs.y)

    def __sub__(self, rhs):
        if isinstance(rhs, (float, int)):
            return ImVec2(self.x - rhs, self.y - rhs)
        return ImVec2(self.x - rhs.x, self.y - rhs.y)

    def __truediv__(self, rhs):
        if isinstance(rhs, (float, int)):
            return ImVec2(self.x / rhs, self.y / rhs)
        return ImVec2(self.x / rhs.x, self.y / rhs.y)

    def __mul__(self, rhs):
        if isinstance(rhs, (float, int)):
            return ImVec2(self.x * rhs, self.y * rhs)
        return ImVec2(self.x * rhs.x, self.y * rhs.y)

    def __abs__(self):
        return ImVec2(abs(self.x), abs(self.y))


#class ImVec4(object):
#    def __init__(self, x=0.0, y=0.0, z=0.0, w=0.0):
#        self.x = x
#        self.y = y
#        self.z = z
#        self.w = w
#
#    def __iter__(self):
#        return iter((self.x, self.y, self.z, self.w))
#
#
#class ImColor(object):
#    def __init__(self, r=0.0, g=0.0, b=0.0, a=0.0):
#        self.r = r
#        self.g = g
#        self.b = b
#        self.a = a
#
#    def __iter__(self):
#        return iter((self.r, self.g, self.b, self.a))


class ImRect(__ImRect__):
    #def __init__(self, min=None, max=None):
    #    self.min = min if min is not None else ImVec2(0.0, 0.0)
    #    self.max = max if max is not None else ImVec2(0.0, 0.0)
    #
    #def __iter__(self):
    #    return iter((self.min, self.max))

    def contains(self, pos: ImVec2):
        return self.min.x <= pos.x <= self.max.x and self.min.y <= pos.y <= self.max.y

    @property
    def center(self):
        return ImVec2((self.min.x + self.max.x) * 0.5, (self.min.y + self.max.y) * 0.5)

    @property
    def size(self):
        return ImVec2(self.max.x - self.min.x, self.max.y - self.min.y)

    @property
    def width(self):
        return self.max.x - self.min.x

    @property
    def height(self):
        return self.max.y - self.min.y

    @property
    def area(self):
        return (self.max.x - self.min.x) * (self.max.y - self.min.y)

    @property
    def tl(self):
        return self.min

    @property
    def tr(self):
        return ImVec2(self.max.x, self.min.y)

    @property
    def bl(self):
        return ImVec2(self.min.x, self.max.y)

    @property
    def br(self):
        return self.max

    def __mul__(self, rhs):
        assert isinstance(rhs, (float, int))
        return ImRect(self.min * rhs, self.max * rhs)

    def expand(self, size):
        if isinstance(size, float):
            size = ImVec2(size, size)
        return ImRect(self.min - size, self.max + size)

%}

%fragment("PySequenceHelpers", "header") {
    template<typename T>
    void AssignAuto(T*& ptr, T& value) { ptr = &value; }
    template<typename T1, typename T2>
    void AssignAuto(T1& ptr, T2& value) { ptr = value; }

    static int PySequence_RequireLength(PyObject* seq, int length)
    {
        if (!PySequence_Check(seq))
            SWIG_exception_fail(SWIG_TypeError, "expected a sequence of numbers.");
        if (PySequence_Length(seq) != length)
            SWIG_exception_fail(SWIG_ValueError, "expected a sequence of numbers.");
        return SWIG_OK;
fail:
        return SWIG_ERROR;
    }

    template<typename T>
    int PySequence_GetNumber(PyObject* seq, int i, T* out)
    {
        swig::SwigVar_PyObject element(PySequence_GetItem(seq, i));
        if (!PyNumber_Check(element))
            SWIG_exception_fail(SWIG_TypeError, "expected a sequence of numbers.");
        *out = (T)PyFloat_AsDouble(swig::SwigVar_PyObject(PyNumber_Float(element)));
        return SWIG_OK;
fail:
        return SWIG_ERROR;
    }

    PyObject* ConstructImVec2(double x, double y)
    {
        swig::SwigVar_PyObject module = PyImport_ImportModule("imgui");
        swig::SwigVar_PyObject ctor = PyObject_GetAttrString(module, "ImVec2");
        swig::SwigVar_PyObject args = Py_BuildValue("(dd)", x, y);
        return PyObject_CallObject(ctor, args);
    }

    PyObject* ConstructImVec4(double x, double y, double z, double w)
    {
        swig::SwigVar_PyObject module = PyImport_ImportModule("imgui");
        swig::SwigVar_PyObject ctor = PyObject_GetAttrString(module, "ImVec4");
        swig::SwigVar_PyObject args = Py_BuildValue("(dddd)", x, y, z, w);
        return PyObject_CallObject(ctor, args);
    }

    PyObject* ConstructImColor(double r, double g, double b, double a = 1.0)
    {
        swig::SwigVar_PyObject module = PyImport_ImportModule("imgui");
        swig::SwigVar_PyObject ctor = PyObject_GetAttrString(module, "ImColor");
        swig::SwigVar_PyObject args = Py_BuildValue("(dddd)", r, g, b, a);
        return PyObject_CallObject(ctor, args);
    }

    PyObject* ConstructImRect(double min_x, double min_y, double max_x, double max_y)
    {
        swig::SwigVar_PyObject module = PyImport_ImportModule("imgui");
        swig::SwigVar_PyObject ctor = PyObject_GetAttrString(module, "ImRect");
        return PyObject_CallObject(ctor, swig::SwigVar_PyObject(Py_BuildValue("(OO)", ConstructImVec2(min_x, min_y), ConstructImVec2(max_x, max_y))));
    }
}

%fragment("ToRef", "header") {
    template<typename T>
    T& ToRef(T& value) { return value; }
    template<typename T>
    T& ToRef(T* value) { return *value; }
}

%extend ImColor
{
    ImU32 ToU32() { return *$self; }
};

%ignore ImRect;
%ignore ImColor;

%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) ImVec4, ImRect, ImColor, ImU32 col, ImU32 color "$1 = PySequence_Check($input) && PySequence_Length($input) == 4;";
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) ImVec4*, const ImVec4*, ImRect*, const ImRect*, ImColor*, const ImColor*  "$1 = $1 = $input == Py_None || (PySequence_Check($input) && PySequence_Length($input) == 4);";

// Map two-item primitives like ImVec2
%define IMGUI_MAP_PRIMITIVE_TYPE2(CTYPE, PYTYPE, x, y)
%ignore CTYPE;
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) CTYPE,        CTYPE& "$1 = PySequence_Check($input) && PySequence_Length($input) == 2;";
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) CTYPE*, const CTYPE* "$1 = (PySequence_Check($input) && PySequence_Length($input) == 2) || $input == Py_None;";
%typemap(out, fragment="ToRef") CTYPE, CTYPE*, CTYPE&
{
    $result = Construct ## PYTYPE(ToRef($1).x, ToRef($1).y);
}
%typemap(in, fragment="PySequenceHelpers") CTYPE (CTYPE tmp)
{
    if (PySequence_RequireLength($input, 2) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 0, &tmp.x) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 1, &tmp.y) != SWIG_OK) goto fail;
    AssignAuto($1, tmp);
}
%apply CTYPE { CTYPE& };
%typemap(in, fragment="PySequenceHelpers") CTYPE* (CTYPE tmp)
{
    if ($input != Py_None)
    {
        if (PySequence_RequireLength($input, 2) != SWIG_OK) goto fail;
        if (PySequence_GetNumber($input, 0, &tmp.x) != SWIG_OK) goto fail;
        if (PySequence_GetNumber($input, 1, &tmp.y) != SWIG_OK) goto fail;
    }
    AssignAuto($1, tmp);
}
%enddef

// Map two-item primitives like ImVec4
%define IMGUI_MAP_PRIMITIVE_TYPE4(CTYPE, PYTYPE, x, y, z, w)
%ignore CTYPE;
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) CTYPE,        CTYPE& "$1 = PySequence_Check($input) && PySequence_Length($input) == 4;";
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) CTYPE*, const CTYPE* "$1 = (PySequence_Check($input) && PySequence_Length($input) == 4) || $input == Py_None;";
%typemap(out, fragment="ToRef") CTYPE, CTYPE*, CTYPE&
{
    $result = Construct ## PYTYPE(ToRef($1).x, ToRef($1).y, ToRef($1).z, ToRef($1).w);
}
%typemap(in, fragment="PySequenceHelpers") CTYPE (CTYPE tmp)
{
    if (PySequence_RequireLength($input, 4) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 0, &tmp.x) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 1, &tmp.y) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 2, &tmp.z) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 3, &tmp.w) != SWIG_OK) goto fail;
    AssignAuto($1, tmp);
}
%apply CTYPE { CTYPE& };
%typemap(in, fragment="PySequenceHelpers") CTYPE* (CTYPE tmp)
{
    if ($input != Py_None)
    {
        if (PySequence_RequireLength($input, 4) != SWIG_OK) goto fail;
        if (PySequence_GetNumber($input, 0, &tmp.x) != SWIG_OK) goto fail;
        if (PySequence_GetNumber($input, 1, &tmp.y) != SWIG_OK) goto fail;
        if (PySequence_GetNumber($input, 2, &tmp.z) != SWIG_OK) goto fail;
        if (PySequence_GetNumber($input, 3, &tmp.w) != SWIG_OK) goto fail;
    }
    AssignAuto($1, tmp);
}
%enddef

// Map two-item primities containing two-item primitives like ImRect
%define IMGUI_MAP_PRIMITIVE_TYPE2x2(CTYPE, PYTYPE_OUTTER, Min_x, Min_y, Max_x, Max_y)
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) CTYPE,        CTYPE& "$1 = PySequence_Check($input) && PySequence_Length($input) == 2;";
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) CTYPE*, const CTYPE* "$1 = (PySequence_Check($input) && PySequence_Length($input) == 2) || $input == Py_None;";
%typemap(out, fragment="ToRef") CTYPE, CTYPE*, CTYPE&
{
    $result = Construct ## PYTYPE_OUTTER(ToRef($1).Min_x, ToRef($1).Min_y, ToRef($1).Max_x, ToRef($1).Max_y);
}
%typemap(in, fragment="PySequenceHelpers") CTYPE (CTYPE tmp)
{
    if (PySequence_RequireLength($input, 2) != SWIG_OK) goto fail;
    swig::SwigVar_PyObject min(PySequence_GetItem($input, 0));
    swig::SwigVar_PyObject max(PySequence_GetItem($input, 1));
    if (PySequence_RequireLength(min, 2) != SWIG_OK) goto fail;
    if (PySequence_RequireLength(max, 2) != SWIG_OK) goto fail;
    if (PySequence_GetNumber(min, 0, &tmp.Min_x) != SWIG_OK) goto fail;
    if (PySequence_GetNumber(min, 1, &tmp.Min_y) != SWIG_OK) goto fail;
    if (PySequence_GetNumber(max, 0, &tmp.Max_x) != SWIG_OK) goto fail;
    if (PySequence_GetNumber(max, 1, &tmp.Max_y) != SWIG_OK) goto fail;
    AssignAuto($1, tmp);
}
%apply CTYPE { CTYPE& };
%typemap(in, fragment="PySequenceHelpers") CTYPE* (CTYPE tmp)
{
    if ($input != Py_None)
    {
        if (PySequence_RequireLength($input, 2) != SWIG_OK) goto fail;
        swig::SwigVar_PyObject min(PySequence_GetItem($input, 0));
        swig::SwigVar_PyObject max(PySequence_GetItem($input, 1));
        if (PySequence_RequireLength(min, 2) != SWIG_OK) goto fail;
        if (PySequence_RequireLength(max, 2) != SWIG_OK) goto fail;
        if (PySequence_GetNumber(min, 0, &tmp.Min_x) != SWIG_OK) goto fail;
        if (PySequence_GetNumber(min, 1, &tmp.Min_y) != SWIG_OK) goto fail;
        if (PySequence_GetNumber(max, 0, &tmp.Max_x) != SWIG_OK) goto fail;
        if (PySequence_GetNumber(max, 1, &tmp.Max_y) != SWIG_OK) goto fail;
    }
    AssignAuto($1, tmp);
}
%enddef

IMGUI_MAP_PRIMITIVE_TYPE2(ImVec2, ImVec2, x, y)
IMGUI_MAP_PRIMITIVE_TYPE4(ImVec4, ImVec4, x, y, z, w)
IMGUI_MAP_PRIMITIVE_TYPE2x2(ImRect, ImRect, Min.x, Min.y, Max.x, Max.y)
IMGUI_MAP_PRIMITIVE_TYPE4(ImColor, ImColor, Value.x, Value.y, Value.z, Value.w)

%ignore ImGui::PushStyleColor(ImGuiCol idx, ImU32 col);
%typemap(typecheck, precedence=SWIG_TYPECHECK_FLOAT_ARRAY) ImU32 col, ImU32 color "$1 = PySequence_Check($input) && PySequence_Length($input) == 4;";
%typemap(out) ImU32 col, ImU32 color {
    ImColor tmp = $1;
    $result = ConstructImColor(tmp.Value.x, tmp.Value.y, tmp.Value.z, tmp.Value.w);
}
%typemap(in, fragment="PySequenceHelpers") ImU32 col, ImU32 color {
    ImColor tmp;
    if (PySequence_RequireLength($input, 4) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 0, &tmp.Value.x) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 1, &tmp.Value.y) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 2, &tmp.Value.z) != SWIG_OK) goto fail;
    if (PySequence_GetNumber($input, 3, &tmp.Value.w) != SWIG_OK) goto fail;
    $1 = tmp;
}
