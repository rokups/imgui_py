%module implot
%import "../imgui_py/imgui.i"
%{
#include <string.h>
#include <stdarg.h>
#include "implot.h"
#include "implot_internal.h"
#include "numpy/arrayobject.h"
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

%include "../imconfig.h"
%include "implot.h"

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

%define IMPLOT_TYPE_INSTANTIATIONS(T)
%inline %{
void PlotLine(const char* label_id, const T* values, int count, double xscale=1, double xstart=0, ImPlotLineFlags flags=0, int offset=0)
{ ImPlot::PlotLine(label_id, values, count, xscale, xstart, flags, offset); }
void PlotLineXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, ImPlotLineFlags flags=0, int offset=0)
{ assert(xs_count == ys_count); ImPlot::PlotLine(label_id, xs, ys, xs_count, flags, offset); }

void PlotScatter(const char* label_id, const T* values, int count, double xscale=1, double xstart=0, ImPlotScatterFlags flags=0, int offset=0)
{ ImPlot::PlotScatter(label_id, values, count, xscale, xstart, flags, offset); }
void PlotScatterXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, ImPlotScatterFlags flags=0, int offset=0)
{ assert(xs_count == ys_count); ImPlot::PlotScatter(label_id, xs, ys, ys_count, flags, offset); }

void PlotStairs(const char* label_id, const T* values, int count, double xscale=1, double xstart=0, ImPlotStairsFlags flags=0, int offset=0)
{ ImPlot::PlotStairs(label_id, values, count, xscale, xstart, flags, offset); }
void PlotStairsXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, ImPlotStairsFlags flags=0, int offset=0)
{ assert(xs_count == ys_count); ImPlot::PlotStairs(label_id, xs, ys, ys_count, flags, offset); }

void PlotShaded(const char* label_id, const T* values, int count, double yref=0, double xscale=1, double xstart=0, ImPlotShadedFlags flags=0, int offset=0)
{ ImPlot::PlotShaded(label_id, values, count, yref, xscale, xstart, flags, offset); }
void PlotShadedXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, double yref=0, ImPlotShadedFlags flags=0, int offset=0)
{ assert(xs_count == ys_count); ImPlot::PlotShaded(label_id, xs, ys, ys_count, yref, flags, offset); }
void PlotShadedXYY(const char* label_id, const T* xs, int xs_count, const T* ys1, int ys1_count, const T* ys2, int ys2_count, ImPlotShadedFlags flags=0, int offset=0)
{ assert(xs_count == ys1_count); assert(xs_count == ys2_count); ImPlot::PlotShaded(label_id, xs, ys1, ys2, xs_count, flags, offset); }

void PlotBars(const char* label_id, const T* values, int count, double bar_size=0.67, double shift=0, ImPlotBarsFlags flags=0, int offset=0)
{ ImPlot::PlotBars(label_id, values, count, bar_size, shift, flags, offset); }
void PlotBarsXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, double bar_size, ImPlotBarsFlags flags=0, int offset=0)
{ assert(xs_count == ys_count); ImPlot::PlotBars(label_id, xs, ys, ys_count, bar_size, flags, offset); }

//void PlotBarGroups(const char* const label_ids[], const T* values, int item_count, int group_count, double group_size=0.67, double shift=0, ImPlotBarGroupsFlags flags=0);

void PlotErrorBars(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, const T* err, int err_count, ImPlotErrorBarsFlags flags=0, int offset=0)
{ assert(xs_count == ys_count); assert(xs_count == err_count); ImPlot::PlotErrorBars(label_id, xs, ys, err, ys_count, flags, offset); }
void PlotErrorBars2(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, const T* neg, int neg_count, const T* pos, int pos_count, ImPlotErrorBarsFlags flags=0, int offset=0)
{ assert(xs_count == ys_count); assert(xs_count == neg_count); assert(xs_count == pos_count); ImPlot::PlotErrorBars(label_id, xs, ys, neg, pos, pos_count, flags, offset); }

void PlotStems(const char* label_id, const T* values, int count, double ref=0, double scale=1, double start=0, ImPlotStemsFlags flags=0, int offset=0)
{ ImPlot::PlotStems(label_id, values, count, ref, scale, start, flags, offset); }
void PlotStemsXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, double ref=0, ImPlotStemsFlags flags=0, int offset=0)
{ assert(xs_count == ys_count); ImPlot::PlotStems(label_id, xs, ys, ys_count, ref, flags, offset); }

void PlotInfLines(const char* label_id, const T* values, int count, ImPlotInfLinesFlags flags=0, int offset=0)
{ ImPlot::PlotInfLines(label_id, values, count, flags, offset); }

//IMPLOT_TMP void PlotPieChart(const char* const label_ids[], const T* values, int count, double x, double y, double radius, const char* label_fmt="%.1f", double angle0=90, ImPlotPieChartFlags flags=0);

void PlotHeatmap(const char* label_id, const T* values, int count, int rows, int cols, double scale_min=0, double scale_max=0, const char* label_fmt="%.1f", const ImPlotPoint* bounds_min=0, const ImPlotPoint* bounds_max=0, ImPlotHeatmapFlags flags=0)
{
    assert(count == rows * cols);
    ImPlotPoint bounds_min_default(0,0);
    ImPlotPoint bounds_max_default(1,1);
    if (bounds_min == 0)
        bounds_min = &bounds_min_default;
    if (bounds_max == 0)
        bounds_max = &bounds_max_default;
    ImPlot::PlotHeatmap(label_id, values, rows, cols, scale_min, scale_max, label_fmt, *bounds_min, *bounds_max, flags);
}

double PlotHistogram(const char* label_id, const T* values, int count, int bins=ImPlotBin_Sturges, double bar_scale=1.0, const ImPlotRange* range=0, ImPlotHistogramFlags flags=0)
{
    const ImPlotRange range_default;
    if (range == 0)
        range = &range_default;
    return ImPlot::PlotHistogram(label_id, values, count, bins, bar_scale, *range, flags);
}

double PlotHistogram2D(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, int x_bins=ImPlotBin_Sturges, int y_bins=ImPlotBin_Sturges, const ImPlotRect* range=0, ImPlotHistogramFlags flags=0)
{
    assert(xs_count == ys_count);
    const ImPlotRect range_default;
    if (range == 0)
        range = &range_default;
    return ImPlot::PlotHistogram2D(label_id, xs, ys, ys_count, x_bins, y_bins, *range, flags);
}

void PlotDigital(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, ImPlotDigitalFlags flags=0, int offset=0)
{ ImPlot::PlotDigital(label_id, xs, ys, ys_count, flags, offset); }

//IMPLOT_API void PlotImage(const char* label_id, ImTextureID user_texture_id, const ImPlotPoint& bounds_min, const ImPlotPoint& bounds_max, const ImVec2& uv0=ImVec2(0,0), const ImVec2& uv1=ImVec2(1,1), const ImVec4& tint_col=ImVec4(1,1,1,1), ImPlotImageFlags flags=0);
%}
%enddef

IMPLOT_TYPE_INSTANTIATIONS(signed char);
IMPLOT_TYPE_INSTANTIATIONS(signed short);
IMPLOT_TYPE_INSTANTIATIONS(signed int);
IMPLOT_TYPE_INSTANTIATIONS(signed long long);
IMPLOT_TYPE_INSTANTIATIONS(unsigned char);
IMPLOT_TYPE_INSTANTIATIONS(unsigned short);
IMPLOT_TYPE_INSTANTIATIONS(unsigned int);
IMPLOT_TYPE_INSTANTIATIONS(unsigned long long);
IMPLOT_TYPE_INSTANTIATIONS(float);
IMPLOT_TYPE_INSTANTIATIONS(double);

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
