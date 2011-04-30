#include "TFWorkCopy.h"

namespace M4D {
namespace GUI {
	
TFWorkCopy::TFWorkCopy(TFFunctionInterface::Ptr function):
	data_(function),
	coords_(function->getDimension()),
	sizes_(function->getDimension() + 1),
	zoom_(function->getDimension()),
	changed_(true),
	histogramChanged_(true),
	histogramEnabled_(false){
}

TFWorkCopy::~TFWorkCopy(){}

TFFunctionInterface::Ptr TFWorkCopy::getFunction(){

	return data_;
}

TF::Size TFWorkCopy::getDimension(){

	return data_->getDimension();
}

void TFWorkCopy::setDataStructure(const std::vector<TF::Size>& dataStructure){
		
	data_->resize(dataStructure);

	for(TF::Size i = 1; i <= dataStructure.size(); ++i)	computeZoom_(i, zoom_.zoom[i], zoom_.center[i]);
}

//---save-&-load---

void TFWorkCopy::save(TF::XmlWriterInterface* writer){

	writer->writeStartElement("WorkCopy");
			
		writer->writeAttribute("MaxZoom", TF::convert<float, std::string>(zoom_.max));
		writer->writeAttribute("HistLogBase", TF::convert<long double, std::string>(hist_.logBase()));

		for(TF::Size i = 0; i <= data_->getDimension(); ++i)
		{
			writer->writeStartElement("ZoomProperty");
				writer->writeAttribute("Dimension", TF::convert<TF::Size, std::string>(i));
				writer->writeAttribute("Zoom", TF::convert<float, std::string>(zoom_.zoom[i]));
				writer->writeAttribute("Center", TF::convert<float, std::string>(zoom_.center[i]));
			writer->writeEndElement();
		}

	writer->writeEndElement();

	data_->save(writer);
}

void TFWorkCopy::saveFunction(TF::XmlWriterInterface* writer){

	data_->save(writer);
}

bool TFWorkCopy::load(TF::XmlReaderInterface* reader, bool& sideError){

	#ifndef TF_NDEBUG
		std::cout << "Loading work copy..." << std::endl;
	#endif

	sideError = false;

	if(reader->readElement("WorkCopy"))
	{				
		float maxZoom = TF::convert<std::string, float>(reader->readAttribute("MaxZoom"));
		long double logBase = TF::convert<std::string, long double>(reader->readAttribute("HistLogBase"));
		if(maxZoom < 1.0f)
		{
			maxZoom = zoom_.max;
			sideError = true;
		}
		zoom_.max = maxZoom;
		if(logBase <= 1.0f)
		{
			sideError = true;
		}
		else hist_.setLogBase(logBase);

		for(TF::Size i = 0; i <= data_->getDimension(); ++i)
		{
			if(reader->readElement("ZoomProperty"))
			{
				float zoom = TF::convert<std::string, float>(reader->readAttribute("Zoom"));
				float center = TF::convert<std::string, float>(reader->readAttribute("Center"));

				if(zoom < 1.0f || zoom > maxZoom)
				{
					zoom = zoom_.zoom[i];
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

bool TFWorkCopy::loadFunction(TF::XmlReaderInterface* reader){

	return data_->load(reader);
}

//---changes---

bool TFWorkCopy::changed(){

	return changed_;
}

bool TFWorkCopy::histogramChanged(){

	if(histogramChanged_)
	{
		histogramChanged_ = false;
		return true;
	}
	return false;
}

void TFWorkCopy::forceUpdate(const bool updateHistogram){

	changed_ = true;
	if(updateHistogram) histogramChanged_ = true;
}

//---histogram---

void TFWorkCopy::setHistogram(const TF::Histogram::Ptr histogram){

	histogram_ = histogram;
	computeZoom_(histogramIndex, zoom_.zoom[histogramIndex], zoom_.center[histogramIndex]);
}

void TFWorkCopy::setHistogramEnabled(bool value){

	if(histogramEnabled_ != value) histogramChanged_ = true;
	histogramEnabled_ = value;
}

bool TFWorkCopy::histogramEnabled(){

	return histogramEnabled_;
}

void TFWorkCopy::increaseHistogramLogBase(const long double increment){

	if(histogram_)
	{
		hist_.setLogBase(hist_.logBase()*(2.0*increment));
		histogramChanged_ = true;
	}
}
void TFWorkCopy::decreaseHistogramLogBase(const long double increment){

	if(histogram_)
	{
		hist_.setLogBase(hist_.logBase()/(2.0*increment));
		histogramChanged_ = true;
	}
}

//---getters---

TF::Color TFWorkCopy::getColor(const TF::Coordinates& coords){

	tfAssert(coords.size() == data_->getDimension());
	
	TF::Size count = 0;
	TF::Color areaColor = getColorFromZoomedArea_(coords, count, false);

	return areaColor/count;
}

TF::Color TFWorkCopy::getRGBfColor(const TF::Coordinates& coords){

	tfAssert(coords.size() == data_->getDimension());
	
	TF::Size count = 0;
	TF::Color areaColor = getColorFromZoomedArea_(coords, count, true);
	areaColor /= count;
	areaColor.alpha = 1;

	return areaColor;
}

TF::Color TFWorkCopy::getColorFromZoomedArea_(const TF::Coordinates& coords,
								  TF::Size& count,
								  const bool& RGBf,
								  TF::Size dimension){
	
	tfAssert(coords[dimension - 1] >= 0 && coords[dimension - 1] < (int)sizes_[dimension]);

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = coords[dimension - 1]*ratio + zoom_.offset[dimension];
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
		else result += getColorFromZoomedArea_(coords, count, RGBf, dimension + 1);	//recurse
	}
	tfAssert(count > 0);
	return result;
}

float TFWorkCopy::getHistogramValue(const int index){

	tfAssert(index >= 0 && index < (int)sizes_[histogramIndex]);

	if(!histogramEnabled_ || !histogram_) return -1;

	float ratio = histogram_->size()/(sizes_[histogramIndex]*zoom_.zoom[histogramIndex]);
	float indexBase = index*ratio + zoom_.offset[histogramIndex];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	float result = 0;
	int count = 0;
	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)histogram_->size())
		{
			result += hist_.getExpLogValue((*histogram_)[i]);
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

//---setters---

void TFWorkCopy::setComponent1(const TF::Coordinates& coords, const float value){

	tfAssert(coords.size() == data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	setComponentToZoomedArea_(coords, Component1, correctedValue);

	changed_ = true;
}

void TFWorkCopy::setComponent2(const TF::Coordinates& coords, const float value){

	tfAssert(coords.size() == data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	setComponentToZoomedArea_(coords, Component2, correctedValue);

	changed_ = true;
}

void TFWorkCopy::setComponent3(const TF::Coordinates& coords, const float value){

	tfAssert(coords.size() == data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	setComponentToZoomedArea_(coords, Component3, correctedValue);

	changed_ = true;
}

void TFWorkCopy::setAlpha(const TF::Coordinates& coords, const float value){

	tfAssert(coords.size() == data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	setComponentToZoomedArea_(coords, Alpha, correctedValue);

	changed_ = true;
}

void TFWorkCopy::setComponentToZoomedArea_(const TF::Coordinates& coords,
										   const Component& component,
										   const float& value,
										   TF::Size dimension){

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = coords[dimension - 1]*ratio + zoom_.offset[dimension];
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
		else setComponentToZoomedArea_(coords, component, value, dimension + 1);	//recurse
	}
}

//---size---

void TFWorkCopy::resize(const TF::Size dimension, const TF::Size size){
	
	tfAssert(dimension <= data_->getDimension());

	sizes_[dimension] = size;
}

void TFWorkCopy::resize(const std::vector<TF::Size>& sizes){

	tfAssert(sizes.size() == data_->getDimension());

	for(TF::Size i = 1; i <= data_->getDimension(); ++i) sizes_[i] = sizes[i - 1];
}

void TFWorkCopy::resizeHistogram(const TF::Size size){

	sizes_[histogramIndex] = size;
}

//---zoom---

void TFWorkCopy::zoom(const TF::Size dimension, const int center, const int stepCount){

	float nextZoom;
	float position = center/(float)sizes_[dimension];

	nextZoom = zoom_.zoom[dimension] + stepCount;
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;
	if(nextZoom < 1) nextZoom = 1;
	if(nextZoom == zoom_.zoom[dimension]) return;

	computeZoom_(dimension, nextZoom, position);
}

void TFWorkCopy::move(const std::vector<int>& increments){

	float position;
	for(TF::Size i = 1; i <= data_->getDimension(); ++i)
	{
		if(zoom_.zoom[i] == 1) continue;

		position = (sizes_[i]/2.0f + increments[i-1])/(float)sizes_[i];
		computeZoom_(i, zoom_.zoom[i], position);	
	}
}

void TFWorkCopy::zoomHistogram(const int center, const int stepCount){

	zoom(histogramIndex, center, stepCount);
}

void TFWorkCopy::moveHistogram(const int increment){

	if(zoom_.zoom[histogramIndex] == 1) return;

	float position = (sizes_[histogramIndex]/2.0f + increment)/(float)sizes_[histogramIndex];
	computeZoom_(histogramIndex, zoom_.zoom[histogramIndex], position);	
}

void TFWorkCopy::computeZoom_(const TF::Size dimension, const float nextZoom, const float center){

	if(dimension == histogramIndex && !histogram_) return;

	float domain;
	if(dimension == histogramIndex) domain = histogram_->size();
	else domain = data_->getDomain(dimension);

	float ratio = domain/zoom_.zoom[dimension];

	float radius = (domain/nextZoom)/2.0f;
	float offesetInc = zoom_.offset[dimension] + center*ratio - radius;

	float maxOffset = domain - 2.0f*radius;		
	if(offesetInc < 0.0f) offesetInc = 0.0f;
	if(offesetInc > maxOffset) offesetInc = maxOffset;

	zoom_.offset[dimension] = offesetInc;
	zoom_.zoom[dimension] = nextZoom;

	zoom_.center[dimension] = (radius + zoom_.offset[dimension])/domain;
	
	if(dimension == histogramIndex) histogramChanged_ = true;
	else changed_ = true;
}

float TFWorkCopy::getZoom(const TF::Size dimension){

	return zoom_.zoom[dimension];
}

float TFWorkCopy::getZoomCenter(const TF::Size dimension){

	return zoom_.center[dimension];
}

float TFWorkCopy::getMaxZoom(){

	return zoom_.max;
}

void TFWorkCopy::setMaxZoom(const float zoom){
	
	zoom_.max = zoom;
}


} // namespace GUI
} // namespace M4D