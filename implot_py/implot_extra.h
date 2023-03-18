#pragma once

#include "imgui.h"
#include "implot.h"

template <typename T> void PlotLineXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, ImPlotLineFlags flags=0, int offset=0)
{
    assert(xs_count == ys_count);
    ImPlot::PlotLine(label_id, xs, ys, ys_count, flags, offset);
}
template <typename T> void PlotScatterXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, ImPlotScatterFlags flags=0, int offset=0)
{
    assert(xs_count == ys_count);
    ImPlot::PlotScatter(label_id, xs, ys, ys_count, flags, offset);
}
template <typename T> void PlotStairsXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, ImPlotStairsFlags flags=0, int offset=0)
{
    assert(xs_count == ys_count);
    ImPlot::PlotStairs(label_id, xs, ys, ys_count, flags, offset);
}
template <typename T> void PlotShadedXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, double yref=0, ImPlotShadedFlags flags=0, int offset=0)
{
    assert(xs_count == ys_count);
    ImPlot::PlotShaded(label_id, xs, ys, ys_count, yref, flags, offset);
}
template <typename T> void PlotBarsXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, double bar_size, ImPlotBarsFlags flags=0, int offset=0)
{
    assert(xs_count == ys_count);
    ImPlot::PlotBars(label_id, xs, ys, ys_count, bar_size, flags, offset);
}
template <typename T> void PlotStemsXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, double ref=0, ImPlotStemsFlags flags=0, int offset=0)
{
    assert(xs_count == ys_count);
    ImPlot::PlotStems(label_id, xs, ys, ys_count, ref, flags, offset);
}
template <typename T> void PlotDigitalXY(const char* label_id, const T* xs, int xs_count, const T* ys, int ys_count, ImPlotDigitalFlags flags=0, int offset=0)
{
    assert(xs_count == ys_count);
    ImPlot::PlotDigital(label_id, xs, ys, ys_count, flags, offset);
}
template <typename T> void PlotLine(const char* label_id, const T* values, int count, double xscale=1, double xstart=0, ImPlotLineFlags flags=0, int offset=0)
{
    ImPlot::PlotLine(label_id, values, count, xscale, xstart, flags, offset);
}
template <typename T> void PlotScatter(const char* label_id, const T* values, int count, double xscale=1, double xstart=0, ImPlotScatterFlags flags=0, int offset=0)
{
    ImPlot::PlotScatter(label_id, values, count, xscale, xstart, flags, offset);
}
template <typename T> void PlotStairs(const char* label_id, const T* values, int count, double xscale=1, double xstart=0, ImPlotStairsFlags flags=0, int offset=0)
{
    ImPlot::PlotStairs(label_id, values, count, xscale, xstart, flags, offset=0);
}
template <typename T> void PlotShaded(const char* label_id, const T* values, int count, double yref=0, double xscale=1, double xstart=0, ImPlotShadedFlags flags=0, int offset=0)
{
    ImPlot::PlotShaded(label_id, values, count, yref, xscale, xstart, flags, offset);
}
template <typename T> void PlotBars(const char* label_id, const T* values, int count, double bar_size=0.67, double shift=0, ImPlotBarsFlags flags=0, int offset=0)
{
    ImPlot::PlotBars(label_id, values, count, bar_size, shift, flags, offset);
}
template <typename T> void PlotStems(const char* label_id, const T* values, int count, double ref=0, double scale=1, double start=0, ImPlotStemsFlags flags=0, int offset=0)
{
    ImPlot::PlotStems(label_id, values, count, ref, scale, start, flags, offset);
}
template <typename T> void PlotInfLines(const char* label_id, const T* values, int count, ImPlotInfLinesFlags flags=0, int offset=0)
{
    ImPlot::PlotInfLines(label_id, values, count, flags, offset);
}
template <typename T> void PlotHeatmap(const char* label_id, const T* values, int rows, int cols, double scale_min=0, double scale_max=0, const char* label_fmt="%.1f", const ImPlotPoint& bounds_min=ImPlotPoint(0,0), const ImPlotPoint& bounds_max=ImPlotPoint(1,1), ImPlotHeatmapFlags flags=0)
{
    ImPlot::PlotHeatmap(label_id, values, rows, cols, scale_min, scale_max, label_fmt, bounds_min, bounds_max, flags);
}
template <typename T> double PlotHistogram(const char* label_id, const T* values, int count, int bins=ImPlotBin_Sturges, double bar_scale=1.0, ImPlotRange range=ImPlotRange(), ImPlotHistogramFlags flags=0)
{
    return ImPlot::PlotHistogram(label_id, values, count, bins, bar_scale, range, flags);
}
