#include "TFWorkCopy.h"

namespace M4D {
namespace GUI {
	
TFWorkCopy::TFWorkCopy(TFFunctionInterface::Ptr function):
	data_(function),
	sizes_(function->getDimension()+1, 0),
	zoom_(function->getDimension()),
	changes_(function->getDimension()),
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

	for(TF::Size i = 1; i <= dataStructure.size(); ++i)	computeZoom_(i, zoom_.zoom[i], 0.5f);
}

//---save-&-load---

void TFWorkCopy::save(TFXmlWriter::Ptr writer){

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
bool TFWorkCopy::load(TFXmlReader::Ptr reader, bool& sideError){

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

	bool error;
	bool ok = data_->load(reader, error);
	sideError = sideError || error;

	return ok;
}

//---changes---

bool TFWorkCopy::component1Changed(const TF::Size dimension){

	if(changes_[dimension - 1].component1)
	{
		changes_[dimension - 1].component1 = false;
		return true;
	}
	return false;
}

bool TFWorkCopy::component2Changed(const TF::Size dimension){

	if(changes_[dimension - 1].component2)
	{
		changes_[dimension - 1].component2 = false;
		return true;
	}
	return false;
}

bool TFWorkCopy::component3Changed(const TF::Size dimension){

	if(changes_[dimension - 1].component3)
	{
		changes_[dimension - 1].component3 = false;
		return true;
	}
	return false;
}

bool TFWorkCopy::alphaChanged(const TF::Size dimension){

	if(changes_[dimension - 1].alpha)
	{
		changes_[dimension - 1].alpha = false;
		return true;
	}
	return false;
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

	for(TF::Size i = 0; i < changes_.size(); ++i)
	{
		changes_[i].setAllChanged();
	}
	if(updateHistogram) histogramChanged_ = true;
}

//---histogram---

void TFWorkCopy::setHistogram(const TF::Histogram::Ptr histogram){

	histogram_ = histogram;
	histogramChanged_ = true;
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

TF::Color TFWorkCopy::getColor(const TF::Size dimension, const int index){

	tfAssert(index >= 0 && index < (int)sizes_[dimension]);
	tfAssert(dimension >= 0 && dimension <= data_->getDimension());

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = index*ratio + zoom_.offset[dimension];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	TF::Color result(0,0,0,0);
	int count = 0;
	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)data_->getDomain(dimension))
		{
			result += data_->getRGBfColor(dimension, i);
			++count;
		}
	}
	result /= count;
	result.alpha = 1;
	tfAssert(count > 0);
	return result;
}

float TFWorkCopy::getComponent1(const TF::Size dimension, const int index){

	tfAssert(index >= 0 && index < (int)sizes_[dimension]);
	tfAssert(dimension >= 0 && dimension <= data_->getDimension());

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = index*ratio + zoom_.offset[dimension];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	float result = 0;
	int count = 0;
	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)data_->getDomain(dimension))
		{
			result += data_->color(dimension, i).component1;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getComponent2(const TF::Size dimension, const int index){

	tfAssert(index >= 0 && index < (int)sizes_[dimension]);
	tfAssert(dimension >= 0 && dimension <= data_->getDimension());

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = index*ratio + zoom_.offset[dimension];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	float result = 0;
	int count = 0;
	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)data_->getDomain(dimension))
		{
			result += data_->color(dimension, i).component2;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getComponent3(const TF::Size dimension, const int index){

	tfAssert(index >= 0 && index < (int)sizes_[dimension]);
	tfAssert(dimension >= 0 && dimension <= data_->getDimension());

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = index*ratio + zoom_.offset[dimension];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	float result = 0;
	int count = 0;
	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)data_->getDomain(dimension))
		{
			result += data_->color(dimension, i).component3;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getAlpha(const TF::Size dimension, const int index){

	tfAssert(index >= 0 && index < (int)sizes_[dimension]);
	tfAssert(dimension >= 0 && dimension <= data_->getDimension());

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = index*ratio + zoom_.offset[dimension];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	float result = 0;
	int count = 0;
	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)data_->getDomain(dimension))
		{
			result += data_->color(dimension, i).alpha;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
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

void TFWorkCopy::setComponent1(const TF::Size dimension, const int index, const float value){

	tfAssert(dimension >= 0 && dimension <= data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = index*ratio + zoom_.offset[dimension];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)data_->getDomain(dimension))
		{
			data_->color(dimension, i).component1 = correctedValue;
		}
	}
	changes_[dimension-1].component1 = true;
}

void TFWorkCopy::setComponent2(const TF::Size dimension, const int index, const float value){

	tfAssert(dimension >= 0 && dimension <= data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = index*ratio + zoom_.offset[dimension];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)data_->getDomain(dimension))
		{
			data_->color(dimension, i).component2 = correctedValue;
		}
	}
	changes_[dimension-1].component2 = true;
}

void TFWorkCopy::setComponent3(const TF::Size dimension, const int index, const float value){

	tfAssert(dimension >= 0 && dimension <= data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = index*ratio + zoom_.offset[dimension];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)data_->getDomain(dimension))
		{
			data_->color(dimension, i).component3 = correctedValue;
		}
	}
	changes_[dimension-1].component3 = true;
}

void TFWorkCopy::setAlpha(const TF::Size dimension, const int index, const float value){

	tfAssert(dimension >= 0 && dimension <= data_->getDimension());

	float correctedValue = value;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	float ratio = data_->getDomain(dimension)/(sizes_[dimension]*zoom_.zoom[dimension]);
	float indexBase = index*ratio + zoom_.offset[dimension];
	float radius = ratio/2.0;
	float bottom = indexBase - radius;
	float top = indexBase + radius;

	for(int i = bottom; i < top; ++i)
	{
		if(i >= 0 && i < (int)data_->getDomain(dimension))
		{
			data_->color(dimension, i).alpha = correctedValue;
		}
	}
	changes_[dimension-1].alpha = true;
}

//---size---

void TFWorkCopy::resize(const TF::Size dimension, const TF::Size size){

	sizes_[dimension] = size;
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

void TFWorkCopy::move(const std::vector<int> increments){

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
	else changes_[dimension-1].setAllChanged();
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