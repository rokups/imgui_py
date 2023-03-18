%module implot
%import "../imgui_py/imgui.i"
%{
#include <string.h>
#include <stdarg.h>
#include "implot.h"
#include "implot_internal.h"
#include "numpy/arrayobject.h"
#include "implot_extra.h"
%}

%rename("%(undercase)s", match$kind="function") "";
%rename("%(undercase)s", match$kind="variable") "";
%rename("%(regex:/^ImPlot(.+)$/\\1/)s", %$isenumitem) "";
%rename("%(regex:/^(ImPlotCond|ImPlotCol|ImPlotStyle)(.+)$/\\1\\2/)s", %$isenumitem) "";

%include "cstring.i"
%include <typemaps.i>
%include "numpy.i"
%init
%{
import_array();
%}

// ImPlot::ColormapSlider
%apply float* INOUT { float* t };
// ImPlot::DragRect
%apply double* INOUT { double* x1, double* y1, double* x2, double* y2, double* x, double* y };

#define IM_FMTARGS(x)
#define IM_FMTLIST(x)

%define TYPEMAP_NUMPY_TYPES(T)
%apply(T* INPLACE_ARRAY1, int DIM1) {
    (const T* values, int count),
    (const T* xs, int xs_count),
    (const T* ys, int ys_count),
    (const T* ys1, int ys1_count),
    (const T* ys2, int ys2_count),
    (const T* err, int err_count),
    (const T* neg, int neg_count),
    (const T* pos, int pos_count)
}
%enddef

TYPEMAP_NUMPY_TYPES(signed char);
TYPEMAP_NUMPY_TYPES(signed short);
TYPEMAP_NUMPY_TYPES(signed int);
TYPEMAP_NUMPY_TYPES(signed long long);
TYPEMAP_NUMPY_TYPES(unsigned char);
TYPEMAP_NUMPY_TYPES(unsigned short);
TYPEMAP_NUMPY_TYPES(unsigned int);
TYPEMAP_NUMPY_TYPES(unsigned long long);
TYPEMAP_NUMPY_TYPES(float);
TYPEMAP_NUMPY_TYPES(double);

// Not supported so far.
%ignore ImPlot::AnnotationV;
%ignore ImPlot::TagXV;
%ignore ImPlot::TagYV;

%ignore ImPlot::Locator_Default;
%ignore ImPlot::Locator_Time;
%ignore ImPlot::Locator_Log10;
%ignore ImPlot::Locator_SymLog;

%include <carrays.i>
%include <pybuffer.i>

/// region row_ratios/col_ratios
%define ARRAY_WITH_DETACHED_SIZE(type, name, size_argnum)
%typemap(argout) type* name "";
%typemap(in) type* name {
    if ($input != Py_None)
    {
        $1 = (type*)alloca(sizeof(type) * arg##size_argnum);
        if (!PySequence_Check($input))
            SWIG_exception_fail(SWIG_TypeError, "expected a sequence of numbers.");
        if (PySequence_Length($input) != arg##size_argnum)
            SWIG_exception_fail(SWIG_TypeError, "specified sequence is of incorrect length.");
        for (int i = 0; i < arg##size_argnum; i++)
        {
            swig::SwigVar_PyObject element(PySequence_GetItem($input, i));
            if (!PyNumber_Check(element))
                SWIG_exception_fail(SWIG_TypeError, "expected a sequence of numbers.");
            $1[i] = PyFloat_AsDouble(swig::SwigVar_PyObject(PyNumber_Float(element)));
        }
    }
    else
        $1 = 0;
}
%typemap(out) type* name "<TODO>";
%typecheck(SWIG_TYPECHECK_POINTER) type* name {
    $1 = $input == Py_None || PySequence_Check($input);
}
%enddef

ARRAY_WITH_DETACHED_SIZE(float, row_ratios, 2)
ARRAY_WITH_DETACHED_SIZE(float, col_ratios, 3)
/// endregion

/// region SetupAxisFormat
%ignore ImPlot::SetupAxisFormat(ImAxis axis, ImPlotFormatter formatter, void* data=NULL);
%ignore _PyAxisFormatter;
%inline
%{
int _PyAxisFormatter(double value, char* buf, int len, void* user_data)
{
	swig::SwigVar_PyObject arg, res;
	PyObject* callable = (PyObject*)user_data;

	if (!PyCallable_Check(callable))
		SWIG_exception_fail(SWIG_TypeError, "expected a callable.");

	arg = Py_BuildValue("(d)", value);
	res = PyObject_CallObject(callable, arg);
	if (!res || res == Py_None)
		return 0;

	if (!PyUnicode_Check(res))
		SWIG_exception_fail(SWIG_TypeError, "expected a string result.");

	if (const char* result = PyUnicode_AsUTF8(res))
		strncpy(buf, result, len);

	fail:
		return 0;
}
void _DecRefAxisFormatter(ImPlotPlot* plt, ImAxis idx)
{
	ImPlotAxis& axis = plt->Axes[idx];
	if (axis.Formatter == &_PyAxisFormatter)
	{
		Py_DecRef((PyObject*)axis.FormatterData);
		axis.Formatter = nullptr;
		axis.FormatterData = nullptr;
	}
}
void _DecRefAxisFormatterAll(ImPlotPlot* plt)
{
	for (ImAxis axis = 0; axis < ImAxis_COUNT; axis++)
		_DecRefAxisFormatter(plt, axis);
}
void SetupAxisFormatter(ImAxis idx, PyObject* callable)
{
    // TODO: We should hold a reference to callable here, but there is no way to reliably free it later. Python code
    //  must ensure callable will remain alive for duration of the frame, or else things crash.
	_DecRefAxisFormatter(ImPlot::GetCurrentPlot(), idx);
	Py_IncRef(callable);
    ImPlot::SetupAxisFormat(idx, &_PyAxisFormatter, callable);
}
%}
/// endregion

%ignore ImPlotAnnotationCollection;
%ignore ImPlotTagCollection;

// Python uses doubles internally so we can just map these primitives to standard imgui primitives without loss of precision.
IMGUI_MAP_PRIMITIVE_TYPE2(ImPlotPoint, ImVec2, x, y)
IMGUI_MAP_PRIMITIVE_TYPE2(ImPlotRange, ImVec2, Min, Max)
IMGUI_MAP_PRIMITIVE_TYPE2x2(ImPlotRect, ImRect, X.Min, Y.Min, X.Max, Y.Max)

// Possibly due to SWIG bug these versions get mapped also as (x, y) argumnets.
%ignore ImPlot::PixelsToPlot(const ImVec2& pix, ImAxis x_axis = IMPLOT_AUTO, ImAxis y_axis = IMPLOT_AUTO);
%ignore ImPlot::PlotToPixels(const ImPlotPoint& plt, ImAxis x_axis = IMPLOT_AUTO, ImAxis y_axis = IMPLOT_AUTO);

// Decrease refcounts if any
%pythonprepend ImPlot::EndPlot() %{
plt = get_current_plot()
%}
%pythonappend ImPlot::EndPlot() %{
_implot_py._dec_ref_axis_formatter_all(plt)
%}

%ignore ImPlot::PlotLine;
%ignore ImPlot::PlotLineG;
%ignore ImPlot::PlotScatter;
%ignore ImPlot::PlotScatterG;
%ignore ImPlot::PlotStairs;
%ignore ImPlot::PlotStairsG;
%ignore ImPlot::PlotShaded;
%ignore ImPlot::PlotShadedG;
%ignore ImPlot::PlotBars;
%ignore ImPlot::PlotBarsG;
%ignore ImPlot::PlotBarGroups;
%ignore ImPlot::PlotErrorBars;
%ignore ImPlot::PlotStems;
%ignore ImPlot::PlotInfLines;
%ignore ImPlot::PlotPieChart;
%ignore ImPlot::PlotHeatmap;
%ignore ImPlot::PlotHistogram;
%ignore ImPlot::PlotHistogram2D;
%ignore ImPlot::PlotDigital;
%ignore ImPlot::PlotDigitalG;
%ignore ImPlot::PlotImage;

%include "../imconfig.h"
%include "implot.h"
%include "implot_extra.h"

#define QUOTE(str) #str
#define EXPAND_AND_QUOTE(str) QUOTE(str)
%define PLOT_TYPE_INSTANTIATIONS(PlotFuncName, BaseName, targn)
%rename(BaseName ## _int8) PlotFuncName<signed char>;
%template() PlotFuncName<signed char>;
%rename(BaseName ## _int16) PlotFuncName<signed short>;
%template() PlotFuncName<signed short>;
%rename(BaseName ## _int32) PlotFuncName<signed int>;
%template() PlotFuncName<signed int>;
%rename(BaseName ## _int64) PlotFuncName<signed long long>;
%template() PlotFuncName<signed long long>;
%rename(BaseName ## _uint8) PlotFuncName<unsigned char>;
%template() PlotFuncName<unsigned char>;
%rename(BaseName ## _uint16) PlotFuncName<unsigned short>;
%template() PlotFuncName<unsigned short>;
%rename(BaseName ## _uint32) PlotFuncName<unsigned int>;
%template() PlotFuncName<unsigned int>;
%rename(BaseName ## _uint64) PlotFuncName<unsigned long long>;
%template() PlotFuncName<unsigned long long>;
%rename(BaseName ## _float32) PlotFuncName<float>;
%template() PlotFuncName<float>;
%rename(BaseName ## _float64) PlotFuncName<double>;
%template() PlotFuncName<double>;
%pythonbegin
%{
def BaseName(*args, **kwargs):
    return globals()[EXPAND_AND_QUOTE(BaseName) + f'_{args[targn].dtype.name}'](*args, **kwargs)
%}
%enddef

PLOT_TYPE_INSTANTIATIONS(PlotLine, plot_line, 1);
PLOT_TYPE_INSTANTIATIONS(PlotScatter, plot_scatter, 1);
PLOT_TYPE_INSTANTIATIONS(PlotStairs, plot_stairs, 1);
PLOT_TYPE_INSTANTIATIONS(PlotShaded, plot_shaded, 1);
PLOT_TYPE_INSTANTIATIONS(PlotBars, plot_bars, 1);
PLOT_TYPE_INSTANTIATIONS(PlotStems, plot_stems, 1);
PLOT_TYPE_INSTANTIATIONS(PlotLineXY, plot_line_xy, 1);
PLOT_TYPE_INSTANTIATIONS(PlotScatterXY, plot_scatter_xy, 1);
PLOT_TYPE_INSTANTIATIONS(PlotStairsXY, plot_stairs_xy, 1);
PLOT_TYPE_INSTANTIATIONS(PlotShadedXY, plot_shaded_xy, 1);
PLOT_TYPE_INSTANTIATIONS(PlotBarsXY, plot_bars_xy, 1);
PLOT_TYPE_INSTANTIATIONS(PlotStemsXY, plot_stems_xy, 1);
PLOT_TYPE_INSTANTIATIONS(PlotDigitalXY, plot_digital_xy, 1);
PLOT_TYPE_INSTANTIATIONS(ImPlot::PlotHistogram2D, plot_histogram_2d, 1);

%ignore ImPlotContext::Plots;
%ignore ImPlotContext::Subplots;
%ignore ImPlotContext::ColorModifiers;
%ignore ImPlotContext::StyleModifiers;
%ignore ImPlotContext::TempDouble1;
%ignore ImPlotContext::TempDouble2;
%ignore ImPlotContext::TempInt1;
%ignore ImPlotContext::AlignmentData;

%ignore ImPlotPlot::Axes;
%ignore ImPlotAxis::SetRange(const ImPlotRange& range);

%include "implot_internal.h"

%extend ImPlotPlot
{
	ImPlotAxis* GetAxis(ImAxis axis)
	{
		return &$self->Axes[axis];
	}
};
