#include <TFWorkCopy.h>


namespace M4D {
namespace GUI {

TFWorkCopy::TFWorkCopy(TFAbstractFunction::Ptr function):
	data_(function),
	xSize_(0),
	ySize_(0),
	component1Changed_(true),
	component2Changed_(true),
	component3Changed_(true),
	alphaChanged_(true),
	histogramChanged_(true),
	histogramEnabled_(false){
}

//---change---

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

//---histogram---

void TFWorkCopy::setHistogram(const TF::Histogram::Ptr histogram){

	histogram_ = histogram;
	setDomain(histogram->size());
	histogramChanged_ = true;
}

void TFWorkCopy::setHistogramEnabled(bool value){

	if(histogramEnabled_ != value) histogramChanged_ = true;
	histogramEnabled_ = value;
}

bool TFWorkCopy::histogramEnabled(){

	return histogramEnabled_;
}

//---getters---

TF::Color TFWorkCopy::getColor(const TF::Size index){

	TF::Color result(0,0,0,0);
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			result += data_->getMappedRGBfColor(currIndex);
			++count;
		}
	}
	result /= count;
	result.alpha = 1;
	tfAssert(count > 0);
	return result;
}

float TFWorkCopy::getComponent1(const TF::Size index){

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			result += ((*data_)[currIndex].component1 - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getComponent2(const TF::Size index){

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			result += ((*data_)[currIndex].component2 - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getComponent3(const TF::Size index){

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			result += ((*data_)[currIndex].component3 - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getAlpha(const TF::Size index){

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			result += ((*data_)[currIndex].alpha - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

float TFWorkCopy::getHistogramValue(const TF::Size index){

	if(!histogramEnabled_ || !histogram_) return -1;

	float result = 0;
	int count = 0;
	int bottom = -zoom_.ratio/2;
	int top = (zoom_.ratio+1)/2;
	int indexBase = index*zoom_.xRatio + zoom_.xOffset + 1;
	int currIndex = indexBase;
	for(int i = bottom; i < top; ++i)
	{
		currIndex = indexBase + i;
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			result += (((*histogram_)[currIndex]/(2.0*histogram_->avarage())) - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

//---setters---

void TFWorkCopy::setComponent1(const TF::Size index, const float value){

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
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			(*data_)[currIndex].component1 = correctedValue;
		}
	}
	component1Changed_ = true;
}

void TFWorkCopy::setComponent2(const TF::Size index, const float value){

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
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			(*data_)[currIndex].component2 = correctedValue;
		}
	}
	component2Changed_ = true;
}

void TFWorkCopy::setComponent3(const TF::Size index, const float value){

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
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			(*data_)[currIndex].component3 = correctedValue;
		}
	}
	component3Changed_ = true;
}

void TFWorkCopy::setAlpha(const TF::Size index, const float value){

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
		if(currIndex >= 0 && currIndex < (int)data_->getDomain())
		{
			(*data_)[currIndex].alpha = correctedValue;
		}
	}
	alphaChanged_ = true;
}

//---size---

void TFWorkCopy::resize(const TF::Size xSize, const TF::Size ySize){

	xSize_ = xSize;
	ySize_ = ySize;	
	computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
}

//---zoom---

void TFWorkCopy::zoomIn(const TF::Size stepCount, const int zoomX, const int zoomY){

	if(zoom_.zoom == zoom_.max) return;

	float nextZoom = zoom_.zoom + stepCount;	
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;

	computeZoom_(nextZoom, zoomX, zoomY);
}

void TFWorkCopy::zoomOut(const TF::Size stepCount, const int zoomX, const int zoomY){

	if(zoom_.zoom == 1) return;

	float nextZoom = zoom_.zoom - stepCount;
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;
	if(nextZoom < 1) nextZoom = 1;

	computeZoom_(nextZoom, zoomX, zoomY);
}
/*
void TFWorkCopy::zoom(const float zoom, const int zoomX, const float zoomY){

	if(zoom == zoom_.zoom && zoomX == zoom_.center.x && zoomY == zoom_.center.y) return;

	float nextZoom = zoom;
	if(nextZoom < 1) nextZoom = 1;
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;

	int xCoord = (zoomX-zoom_.xOffset)/zoom_.xRatio;
	int yCoord = (int)(((zoomY-zoom_.yOffset)*zoom_.zoom)*ySize_);
	computeZoom_(nextZoom, xCoord, yCoord);
}
*/
void TFWorkCopy::move(int xDirectionIncrement, int yDirectionIncrement){

	if(zoom_.zoom == 1) return;

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

TF::Point<float, float> TFWorkCopy::getZoomCenter() const{

	return zoom_.center;
}

void TFWorkCopy::computeZoom_(const float nextZoom, const int zoomX, const int zoomY){

	float relativeZoomedRatioX = (data_->getDomain()/zoom_.zoom)/xSize_;
	float relativeZoomedRatioY = (1/zoom_.zoom)/ySize_;

	float xRadius = (int)(data_->getDomain()/nextZoom/2.0f);
	int xOffesetInc = zoom_.xOffset + (int)(zoomX*relativeZoomedRatioX - xRadius);

	int maxXOffset = (int)(data_->getDomain() - 2*xRadius);
	
	if(xOffesetInc < 0) xOffesetInc = 0;
	if(xOffesetInc > maxXOffset) xOffesetInc = maxXOffset;

	float yRadius = 1/nextZoom/2.0f;
	float yOffesetInc = zoom_.yOffset + zoomY*relativeZoomedRatioY - yRadius;

	float maxYOffset = 1 - 2*yRadius;

	if(yOffesetInc < 0) yOffesetInc = 0;
	if(yOffesetInc > maxYOffset) yOffesetInc = maxYOffset;

	float zoomedDomain = data_->getDomain()/nextZoom;

	zoom_.zoom = nextZoom;
	zoom_.xOffset = xOffesetInc;
	zoom_.yOffset = yOffesetInc;
	zoom_.xRatio = zoomedDomain/xSize_;
	zoom_.ratio = (int)(zoom_.xRatio)+1;
	zoom_.center = TF::Point<float,float>(((zoomedDomain/2.0) + zoom_.xOffset)/data_->getDomain(),
		1.0/zoom_.zoom/2.0 + zoom_.yOffset);
	
	histogramChanged_ = true;
	component1Changed_ = true;
	component2Changed_ = true;
	component3Changed_ = true;
	alphaChanged_ = true;
}

//---update---

TFAbstractFunction::Ptr TFWorkCopy::getFunctionMemento() const{

	return data_->clone();
}

TFAbstractFunction::Ptr TFWorkCopy::getFunction() const{

	return data_;
}

void TFWorkCopy::setDomain(const TF::Size domain){
	
	if(domain == data_->getDomain()) return;
	
	data_->resize(domain);

	if(histogram_ && histogram_->size() != data_->getDomain())
	{
		histogram_ = TF::Histogram::Ptr();
	}

	computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
}

void TFWorkCopy::update(const TFAbstractFunction::Ptr function){

	data_ = function->clone();

	if(histogram_ && histogram_->size() != data_->getDomain())
	{
		data_->resize(histogram_->size());
	}

	computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
}

} // namespace GUI
} // namespace M4D