#include <TFWorkCopy.h>


namespace M4D {
namespace GUI {

TFWorkCopy::TFWorkCopy(const TFSize domain):
	domain_(domain),
	data_(new TFColorMap(domain)),
	xSize_(0),
	ySize_(0),
	component1Changed_(true),
	component2Changed_(true),
	component3Changed_(true),
	alphaChanged_(true),
	histogramChanged_(true),
	histogramEnabled_(false){
}

bool TFWorkCopy::component1Changed(){

	if(component1Changed_)
	{
		component1Changed_ = false;
		return true;
	}
	return false;
}

bool TFWorkCopy::component2Changed(){

	if(component2Changed_)
	{
		component2Changed_ = false;
		return true;
	}
	return false;
}

bool TFWorkCopy::component3Changed(){

	if(component3Changed_)
	{
		component3Changed_ = false;
		return true;
	}
	return false;
}

bool TFWorkCopy::alphaChanged(){

	if(alphaChanged_)
	{
		alphaChanged_ = false;
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

void TFWorkCopy::setHistogram(TFHistogramPtr histogram){

	histogramChanged_ = true;
	histogram_ = histogram;
}

void TFWorkCopy::setHistogramEnabled(bool value){

	if(histogramEnabled_ != value) histogramChanged_ = true;
	histogramEnabled_ = value;
}

bool TFWorkCopy::histogramEnabled(){

	return histogramEnabled_;
}

//---getters---

TFColor TFWorkCopy::getColor(const TFSize index){

	TFColor result(0,0,0,0);
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			result += (*data_)[currIndex];
			++count;
		}
	}
	result /= count;
	result.alpha = 1;
	tfAssert(count > 0);
	return result;
}

float TFWorkCopy::getComponent1(const TFSize index){

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			result += ((*data_)[currIndex].component1 - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getComponent2(const TFSize index){

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			result += ((*data_)[currIndex].component2 - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getComponent3(const TFSize index){

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			result += ((*data_)[currIndex].component3 - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getAlpha(const TFSize index){

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			result += ((*data_)[currIndex].alpha - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getHistogramValue(const TFSize index){

	if(!histogramEnabled_ || !histogram_) return -1;

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset + 1;
	int currIndex = indexBase;
	float maxHistValue = HistogramGetMaxCount(*histogram_);
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			result += (((*histogram_)[currIndex]/maxHistValue) - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

//---setters---

void TFWorkCopy::setComponent1(const TFSize index, const float value){

	float correctedValue = value/zoom_.zoom + zoom_.yOffset;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;

	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			(*data_)[currIndex].component1 = correctedValue;
		}
	}
	component1Changed_ = true;
}

void TFWorkCopy::setComponent2(const TFSize index, const float value){

	float correctedValue = value/zoom_.zoom + zoom_.yOffset;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			(*data_)[currIndex].component2 = correctedValue;
		}
	}
	component2Changed_ = true;
}

void TFWorkCopy::setComponent3(const TFSize index, const float value){

	float correctedValue = value/zoom_.zoom + zoom_.yOffset;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			(*data_)[currIndex].component3 = correctedValue;
		}
	}
	component3Changed_ = true;
}

void TFWorkCopy::setAlpha(const TFSize index, const float value){

	float correctedValue = value/zoom_.zoom + zoom_.yOffset;
	if(correctedValue < 0) correctedValue = 0;
	if(correctedValue > 1) correctedValue = 1;

	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)domain_)
		{
			(*data_)[currIndex].alpha = correctedValue;
		}
	}
	alphaChanged_ = true;
}

//---size---

void TFWorkCopy::resize(const TFSize xSize, const TFSize ySize){

	xSize_ = xSize;
	ySize_ = ySize;	
	computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
}

//---zoom---

void TFWorkCopy::zoomIn(const TFSize stepCount, const int zoomX, const int zoomY){

	if(zoom_.zoom == zoom_.max) return;

	float nextZoom = zoom_.zoom + stepCount;	
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;

	computeZoom_(nextZoom, zoomX, zoomY);
}

void TFWorkCopy::zoomOut(const TFSize stepCount, const int zoomX, const int zoomY){

	if(zoom_.zoom == 1) return;

	float nextZoom = zoom_.zoom - stepCount;
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;
	if(nextZoom < 1) nextZoom = 1;

	computeZoom_(nextZoom, zoomX, zoomY);
}

void TFWorkCopy::zoom(const float zoom, const int zoomX, const float zoomY){

	if(zoom == zoom_.zoom && zoomX == zoom_.center.x && zoomY == zoom_.center.y) return;

	float nextZoom = zoom;
	if(nextZoom < 1) nextZoom = 1;
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;

	int xCoord = (zoomX-zoom_.xOffset)/zoom_.xRatio;
	int yCoord = (int)(((zoomY-zoom_.yOffset)*zoom_.zoom)*ySize_);
	computeZoom_(nextZoom, xCoord, yCoord);
}

void TFWorkCopy::move(int xDirectionIncrement, int yDirectionIncrement){


	int moveX = xSize_/2 + xDirectionIncrement;
	int moveY = ySize_/2 + yDirectionIncrement;

	computeZoom_(zoom_.zoom, moveX, moveY);	
}

float TFWorkCopy::getZoom() const{

	return zoom_.zoom;
}

float TFWorkCopy::getMaxZoom() const{

	return zoom_.max;
}

void TFWorkCopy::setMaxZoom(const float zoom){
	
	zoom_.max = zoom;
}

TFPoint<float, float> TFWorkCopy::getZoomCenter() const{

	return zoom_.center;
}

void TFWorkCopy::computeZoom_(const float nextZoom, const int zoomX, const int zoomY){

	float relativeZoomedRatioX = (domain_/zoom_.zoom)/xSize_;
	float relativeZoomedRatioY = (1/zoom_.zoom)/ySize_;

	int xRadius = (int)(domain_/nextZoom/2.0f);
	int xOffesetInc = zoom_.xOffset + (int)(zoomX*relativeZoomedRatioX) - xRadius;

	int maxXOffset = domain_ - 2*xRadius;
	
	if(xOffesetInc < 0) xOffesetInc = 0;
	if(xOffesetInc > maxXOffset) xOffesetInc = maxXOffset;

	float yRadius = 1/nextZoom/2.0f;
	float yOffesetInc = zoom_.yOffset + zoomY*relativeZoomedRatioY - yRadius;

	float maxYOffset = 1 - 2*yRadius;

	if(yOffesetInc < 0) yOffesetInc = 0;
	if(yOffesetInc > maxYOffset) yOffesetInc = maxYOffset;

	float zoomedDomain = domain_/nextZoom;

	zoom_.zoom = nextZoom;
	zoom_.xOffset = xOffesetInc;
	zoom_.yOffset = yOffesetInc;
	zoom_.xRatio = zoomedDomain/xSize_;
	zoom_.ratio = (int)(zoom_.xRatio)+1;
	zoom_.center = TFPoint<float,float>(((zoomedDomain/2.0) + zoom_.xOffset)/domain_,
		1.0/zoom_.zoom/2.0 + zoom_.yOffset);

	histogramChanged_ = true;
	component1Changed_ = true;
	component2Changed_ = true;
	component3Changed_ = true;
	alphaChanged_ = true;
}

//---update---

void TFWorkCopy::updateFunction(TFAbstractFunction::Ptr function){

	*(function->getColorMap()) = *data_;
}

void TFWorkCopy::update(TFAbstractFunction::Ptr function){

	if(domain_ != function->getDomain())
	{
		domain_ = function->getDomain();
		data_ = TFColorMapPtr(new TFColorMap(domain_));
		computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
	}
	*data_ = *(function->getColorMap());
}

} // namespace GUI
} // namespace M4D