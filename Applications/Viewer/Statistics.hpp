#pragma once


#include "MedV4D/Imaging/ImageTools.h"
#include "MedV4D/Imaging/Histogram.h"

#include "tfw/data/AStatistics.hpp"

class ImageStatistics : public tfw::AStatistics
{
public:
	bool
	hasHistogram() const override
	{
		return true;
	}

	std::pair<float, float>
	getHistogramRange() const override
	{
		auto range = mHistogram.getRange();
		return std::make_pair(float(range.first), float(range.second));
	}

	virtual std::vector<QPointF>
	getHistogramSamples() const override
	{
		auto extremes = mHistogram.minmax();
		std::vector<QPointF> points;
		points.reserve(mHistogram.resolution()[0]);
		float step = float(mHistogram.getRange().second - mHistogram.getRange().first) / mHistogram.resolution()[0];
		float x = 0.0f;//TODO
		for (auto value : mHistogram) {
			points.emplace_back(x, float(value) / extremes.second);
			x += step;
		}
		return points;
	}

	bool
	hasScatterPlot() const override
	{
		return !mGradientScatterPlot.empty();
	}

	std::pair<QRectF, tfw::ScatterPlotData>
	getScatterPlot() const override
	{
		auto range = mGradientScatterPlot.getRange();
		QRectF region(
			range.first.first,
			range.first.second,
			range.second.first - range.first.first,
			range.second.second - range.first.second);

		auto minmax = mGradientScatterPlot.minmax();
		auto resolution = mGradientScatterPlot.resolution();

		tfw::ScatterPlotData data;
		data.size[0] = resolution[0];
		data.size[1] = resolution[1];
		data.buffer.resize(resolution[0] * resolution[1]);

		for (int j = 0; j < resolution[1]; ++j) {
			for (int i = 0; i < resolution[0]; ++i) {
				data.buffer[i + j*resolution[0]] = double(mGradientScatterPlot.data()[i + j*resolution[0]]) / minmax.second;
			}
		}
		return std::make_pair(region, std::move(data));
	}

	M4D::Imaging::Histogram1D<int> mHistogram;
	M4D::Imaging::ScatterPlot2D<int, float> mGradientScatterPlot;
};

