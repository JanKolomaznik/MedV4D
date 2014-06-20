#include "MedV4D/GUI/TF/WorkCopy.h"

namespace M4D {
namespace GUI {

WorkCopy::WorkCopy(TransferFunctionInterface::Ptr function):
	data_(function),
	coords_(function->getDimension()),
	sizes_(function->getDimension()),
	zoom_(function->getDimension()),
	changed_(true),
	histogramChanged_(true),
	histogramEnabled_(false){
}

WorkCopy::~WorkCopy(){}

TransferFunctionInterface::Ptr WorkCopy::getFunction(){

	return data_;
}

TF::Size WorkCopy::getDimension(){

	return data_->getDimension();
}

void WorkCopy::setDataStructure(const std::vector<TF::Size>& dataStructure){

	data_->resize(dataStructure);

	for(TF::Size i = 1; i <= dataStructure.size(); ++i)	computeZoom_(i, zoom_.zoom[i-1], zoom_.center[i-1]);
}

//---save-&-load---

void WorkCopy::save(TF::XmlWriterInterface* writer){

	writer->writeStartElement("WorkCopy");

		writer->writeAttribute("MaxZoom", TF::convert<float, std::string>(zoom_.max));
		writer->writeAttribute("HistLogBase", TF::convert<long double, std::string>(hist_.logBase()));

		for(TF::Size i = 1; i <= data_->getDimension(); ++i)
		{
			writer->writeStartElement("ZoomProperty");
				writer->writeAttribute("Dimension", TF::convert<TF::Size, std::string>(i));
				writer->writeAttribute("Zoom", TF::convert<float, std::string>(zoom_.zoom[i-1]));
				writer->writeAttribute("Center", TF::convert<float, std::string>(zoom_.center[i-1]));
			writer->writeEndElement();
		}

	writer->writeEndElement();

	data_->save(writer);
}

void WorkCopy::saveFunction(TF::XmlWriterInterface* writer){

	data_->save(writer);
}

bool WorkCopy::load(TF::XmlReaderInterface* reader, bool& sideError){

	#ifndef TF_NDEBUG
		std::cout << "Loading work copy..." << std::endl;
	#endif

	sideError = false;

	if(reader->readElement("WorkCopy"))
	{
		float maxZoom = TF::convert<std::string, float>(reader->readAttribute("MaxZoom"));
		if(maxZoom < 1.0f)
		{
			maxZoom = zoom_.max;
			sideError = true;
		}
		zoom_.max = maxZoom;

		long double logBase = TF::convert<std::string, long double>(reader->readAttribute("HistLogBase"));
		if(logBase <= 1.0f)
		{
			sideError = true;
		}
		else hist_.setLogBase(logBase);

		for(TF::Size i = 1; i <= data_->getDimension(); ++i)
		{
			if(reader->readElement("ZoomProperty"))
			{
				float zoom = TF::convert<std::string, float>(reader->readAttribute("Zoom"));
				float center = TF::convert<std::string, float>(reader->readAttribute("Center"));

				if(zoom < 1.0f || zoom > maxZoom)
				{
					zoom = zoom_.zoom[i-1];
					sideError = true;
				}
				computeZoom_(i, zoom, center);
			}
			else
			{
				sideError = true;
				break;
			}
		}

	}
	else sideError = true;

	return data_->load(reader);
}

bool WorkCopy::loadFunction(TF::XmlReaderInterface* reader){

	return data_->load(reader);
}

//---changes---

bool WorkCopy::changed(){

	return changed_;
}

bool WorkCopy::histogramChanged(){

	if(histogramChanged_)
	{
		histogramChanged_ = false;
		return true;
	}
	return false;
}

void WorkCopy::forceUpdate(const bool updateHistogram){

	changed_ = true;
	if(updateHistogram) histogramChanged_ = true;
}

//---histogram---

void WorkCopy::setHistogram(const TF::HistogramInterface::Ptr histogram){

	histogram_ = histogram;
	histogramChanged_ = true;
}

void WorkCopy::setHistogramEnabled(bool value){

	if(histogramEnabled_ != value) histogramChanged_ = true;
	histogramEnabled_ = value;
}

bool WorkCopy::histogramEnabled(){

	return histogramEnabled_;
}

void WorkCopy::increaseHistogramLogBase(const long double increment){

	if(histogram_)
	{
		hist_.setLogBase(hist_.logBase()*(2.0*increment));
		histogramChanged_ = true;
	}
}
void WorkCopy::decreaseHistogramLogBase(const long double increment){

	if(histogram_)
	{
		hist_.setLogBase(hist_.logBase()/(2.0*increment));
		histogramChanged_ = true;
	}
}

void
WorkCopy::setStatistics(M4D::Imaging::Statistics::Ptr aStatistics) {
	mStatistics = aStatistics;
}

//---getters---

TF::Color WorkCopy::getColor(const TF::Coordinates& coords){

	tfAssert(coords.size() == data_->getDimension());

	TF::Size count = 0;
	TF::Color areaColor = getColor_(coords, count, false);

	return areaColor/count;
}

TF::Color WorkCopy::getRGBfColor(const TF::Coordinates& coords){

	tfAssert(coords.size() == data_->getDimension());

	TF::Size count = 0;
	TF::Color areaColor = getColor_(coords, count, true);
	areaColor /= count;
	areaColor.alpha = 1;

	return areaColor;
}

TF::Color WorkCopy::getColor_(const TF::Coordinates& coords,
								  TF::Size& count,
								  const bool& RGBf,
								  TF::Size dimension){

	tfAssert(coords[dimension - 1] >= 0 && coords[dimension - 1] < (int)sizes_[dimension-1]);

	float ratio = data_->getDomain(dimension)/(sizes_[dimension-1]*zoom_.zoom[dimension-1]);
	float indexBase = coords[dimension - 1]*ratio + zoom_.offset[dimension-1];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	TF::Color result(0,0,0,0);
	for(int i = bottom; i < top; ++i)	//zoomed area
	{
		if(i < 0 || i >= (int)data_->getDomain(dimension)) continue;	//out of range

		coords_[dimension - 1] = i;	//fixing index for recurse
		if(dimension == data_->getDimension())	//end of recursion
		{
			if(RGBf) result += data_->getRGBfColor(coords_);
			else result += data_->color(coords_);
			++count;
		}
		else result += getColor_(coords, count, RGBf, dimension + 1);	//recurse
	}
	tfAssert(count > 0);
	return result;
}

float WorkCopy::getHistogramValue(const TF::Coordinates& coords){

	tfAssert(coords.size() == data_->getDimension());

	if(!histogramEnabled_ || !histogram_) return 0;

	TF::Size count = 0;
	float areaValue = getHistogramValue_(coords, count);
	areaValue /= count;

	return areaValue;
}

float WorkCopy::getHistogramValue_(const TF::Coordinates& coords,
								  TF::Size& count,
								  TF::Size dimension){

	tfAssert(coords[dimension - 1] >= 0 && coords[dimension - 1] < (int)sizes_[dimension-1]);

	float ratio = histogram_->getDomain(dimension)/(sizes_[dimension-1]*zoom_.zoom[dimension-1]);
	float indexBase = coords[dimension - 1]*ratio + zoom_.offset[dimension-1];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	float result = 0;
	for(int i = bottom; i < top; ++i)
	{
		if(i < 0 || i >= (int)histogram_->getDomain(dimension)) continue;	//out of range

		coords_[dimension - 1] = i;	//fixing index for recurse
		if(dimension == data_->getDimension())	//end of recursion
		{
			result += hist_.getExpLogValue(histogram_->get(coords_));
			++count;
		}
		else result += getHistogramValue_(coords, count, dimension + 1);	//recurse
	}

	tfAssert(count > 0);
	return result;
}

//---setters---

void WorkCopy::setComponent1(const TF::Coordinates& coords, const float value){

	tfAssert(coords.size() == data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	setComponent_(coords, Component1, correctedValue);

	changed_ = true;
}

void WorkCopy::setComponent2(const TF::Coordinates& coords, const float value){

	tfAssert(coords.size() == data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	setComponent_(coords, Component2, correctedValue);

	changed_ = true;
}

void WorkCopy::setComponent3(const TF::Coordinates& coords, const float value){

	tfAssert(coords.size() == data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	setComponent_(coords, Component3, correctedValue);

	changed_ = true;
}

void WorkCopy::setAlpha(const TF::Coordinates& coords, const float value){

	tfAssert(coords.size() == data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	setComponent_(coords, Alpha, correctedValue);

	changed_ = true;
}

void WorkCopy::setComponent_(const TF::Coordinates& coords,
										   const Component& component,
										   const float& value,
										   TF::Size dimension){

	float ratio = data_->getDomain(dimension)/(sizes_[dimension-1]*zoom_.zoom[dimension-1]);
	float indexBase = coords[dimension - 1]*ratio + zoom_.offset[dimension-1];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	for(int i = bottom; i < top; ++i)	//zoomed area
	{
		if(i < 0 || i >= (int)data_->getDomain(dimension)) continue;	//out of range

		coords_[dimension - 1] = i;	//fixing index for recurse
		if(dimension == data_->getDimension())	//end of recursion
		{
			switch(component)
			{
				case Component1:
				{
					data_->color(coords_).component1 = value;
					break;
				}
				case Component2:
				{
					data_->color(coords_).component2 = value;
					break;
				}
				case Component3:
				{
					data_->color(coords_).component3 = value;
					break;
				}
				case Alpha:
				{
					data_->color(coords_).alpha = value;
					break;
				}
			}
		}
		else setComponent_(coords, component, value, dimension + 1);	//recurse
	}
}

//---size---

void WorkCopy::resize(const TF::Size dimension, const TF::Size size){

	tfAssert(dimension <= data_->getDimension());

	sizes_[dimension-1] = size;
}

void WorkCopy::resize(const std::vector<TF::Size>& sizes){

	tfAssert(sizes.size() == data_->getDimension());

	sizes_ = sizes;
}

//---zoom---

void WorkCopy::zoom(const TF::Size dimension, const int center, const int stepCount){

	float nextZoom;
	float position = center/(float)sizes_[dimension-1];

	nextZoom = zoom_.zoom[dimension-1] + stepCount;
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;
	if(nextZoom < 1) nextZoom = 1;
	if(nextZoom == zoom_.zoom[dimension-1]) return;

	computeZoom_(dimension, nextZoom, position);
}

void WorkCopy::move(const std::vector<int>& increments){

	float position;
	for(TF::Size i = 1; i <= data_->getDimension(); ++i)
	{
		if(zoom_.zoom[i-1] == 1) continue;

		position = (sizes_[i-1]/2.0f + increments[i-1])/(float)sizes_[i-1];
		computeZoom_(i, zoom_.zoom[i-1], position);
	}
}

void WorkCopy::computeZoom_(const TF::Size dimension, const float nextZoom, const float center){

	float domain = data_->getDomain(dimension);

	float ratio = domain/zoom_.zoom[dimension-1];

	float radius = (domain/nextZoom)/2.0f;
	float offesetInc = zoom_.offset[dimension-1] + center*ratio - radius;

	float maxOffset = domain - 2.0f*radius;
	if(offesetInc < 0.0f) offesetInc = 0.0f;
	if(offesetInc > maxOffset) offesetInc = maxOffset;

	zoom_.offset[dimension-1] = offesetInc;
	zoom_.zoom[dimension-1] = nextZoom;

	zoom_.center[dimension-1] = (radius + zoom_.offset[dimension-1])/domain;

	histogramChanged_ = true;
	changed_ = true;
}

float WorkCopy::getZoom(const TF::Size dimension){

	return zoom_.zoom[dimension-1];
}

float WorkCopy::getZoomCenter(const TF::Size dimension){

	return zoom_.center[dimension-1];
}

float WorkCopy::getMaxZoom(){

	return zoom_.max;
}

void WorkCopy::setMaxZoom(const float zoom){

	zoom_.max = zoom;
}


} // namespace GUI
} // namespace M4D
