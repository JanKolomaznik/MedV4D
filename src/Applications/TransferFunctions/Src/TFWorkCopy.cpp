#include <TFWorkCopy.h>


namespace M4D {
namespace GUI {
/*
template<TF::Size dim>
TFWorkCopy<dim>::TFWorkCopy(typename TFAbstractFunction<dim>::Ptr function):
	data_(function),
	xSize_(0),
	ySize_(0),
	histogramChanged_(true),
	histogramEnabled_(false){
}

//---change---

template<TF::Size dim>
bool TFWorkCopy<dim>::component1Changed(TF::Size dimension){

	if(changes_[dimension - 1].component1)
	{
		changes_[dimension - 1].component1 = false;
		return true;
	}
	return false;
}

template<TF::Size dim>
bool TFWorkCopy<dim>::component2Changed(TF::Size dimension){

	if(changes_[dimension - 1].component2)
	{
		changes_[dimension - 1].component2 = false;
		return true;
	}
	return false;
}

template<TF::Size dim>
bool TFWorkCopy<dim>::component3Changed(TF::Size dimension){

	if(changes_[dimension - 1].component3)
	{
		changes_[dimension - 1].component3 = false;
		return true;
	}
	return false;
}

template<TF::Size dim>
bool TFWorkCopy<dim>::alphaChanged(TF::Size dimension){

	if(changes_[dimension - 1].alpha)
	{
		changes_[dimension - 1].alpha = false;
		return true;
	}
	return false;
}

template<TF::Size dim>
bool TFWorkCopy<dim>::histogramChanged(){

	if(histogramChanged_)
	{
		histogramChanged_ = false;
		return true;
	}
	return false;
}

//---histogram---

template<TF::Size dim>
void TFWorkCopy<dim>::setHistogram(const TF::Histogram::Ptr histogram){

	histogram_ = histogram;
	setDomain(histogram->size());
	histogramChanged_ = true;
}

template<TF::Size dim>
void TFWorkCopy<dim>::setHistogramEnabled(bool value){

	if(histogramEnabled_ != value) histogramChanged_ = true;
	histogramEnabled_ = value;
}

template<TF::Size dim>
bool TFWorkCopy<dim>::histogramEnabled(){

	return histogramEnabled_;
}

//---getters---

template<TF::Size dim>
TF::Color TFWorkCopy<dim>::getColor(const TF::Size index, TF::Size dimension){

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
			result += data_->getMappedRGBfColor(currIndex)[dimension];
			++count;
		}
	}
	result /= count;
	result.alpha = 1;
	tfAssert(count > 0);
	return result;
}

template<TF::Size dim>
float TFWorkCopy<dim>::getComponent1(const TF::Size index, TF::Size dimension){

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
			result += ((*data_)[currIndex][dimension].component1 - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

template<TF::Size dim>
float TFWorkCopy<dim>::getComponent2(const TF::Size index, TF::Size dimension){

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
			result += ((*data_)[currIndex][dimension].component2 - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

template<TF::Size dim>
float TFWorkCopy<dim>::getComponent3(const TF::Size index, TF::Size dimension){

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
			result += ((*data_)[currIndex][dimension].component3 - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

template<TF::Size dim>
float TFWorkCopy<dim>::getAlpha(const TF::Size index, TF::Size dimension){

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
			result += ((*data_)[currIndex][dimension].alpha - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

template<TF::Size dim>
float TFWorkCopy<dim>::getHistogramValue(const TF::Size index){

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
		{	//TODO log histogram view
			result += (((*histogram_)[currIndex]/(2.0*histogram_->avarage())) - zoom_.yOffset)*zoom_.zoom;
			++count;
		}
	}
	tfAssert(count > 0);
	return result/count;
}

//---setters---

template<TF::Size dim>
void TFWorkCopy<dim>::setComponent1(const TF::Size index, TF::Size dimension, const float value){

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
			(*data_)[currIndex][dimension].component1 = correctedValue;
		}
	}
	changes_[dimension - 1].component1 = true;
}

template<TF::Size dim>
void TFWorkCopy<dim>::setComponent2(const TF::Size index, TF::Size dimension, const float value){

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
			(*data_)[currIndex][dimension].component2 = correctedValue;
		}
	}
	changes_[dimension - 1].component2 = true;
}

template<TF::Size dim>
void TFWorkCopy<dim>::setComponent3(const TF::Size index, TF::Size dimension, const float value){

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
			(*data_)[currIndex][dimension].component3 = correctedValue;
		}
	}
	changes_[dimension - 1].component3 = true;
}

template<TF::Size dim>
void TFWorkCopy<dim>::setAlpha(const TF::Size index, TF::Size dimension, const float value){

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
			(*data_)[currIndex][dimension].alpha = correctedValue;
		}
	}
	changes_[dimension - 1].alpha = true;
}

//---size---

template<TF::Size dim>
void TFWorkCopy<dim>::resize(const TF::Size xSize, const TF::Size ySize){

	xSize_ = xSize;
	ySize_ = ySize;	
	computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
}

//---zoom---

template<TF::Size dim>
void TFWorkCopy<dim>::zoomIn(const TF::Size stepCount, const int zoomX, const int zoomY){

	if(zoom_.zoom == zoom_.max) return;

	float nextZoom = zoom_.zoom + stepCount;	
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;

	computeZoom_(nextZoom, zoomX, zoomY);
}

template<TF::Size dim>
void TFWorkCopy<dim>::zoomOut(const TF::Size stepCount, const int zoomX, const int zoomY){

	if(zoom_.zoom == 1) return;

	float nextZoom = zoom_.zoom - stepCount;
	if(nextZoom > zoom_.max) nextZoom = zoom_.max;
	if(nextZoom < 1) nextZoom = 1;

	computeZoom_(nextZoom, zoomX, zoomY);
}

template<TF::Size dim>
void TFWorkCopy<dim>::move(int xDirectionIncrement, int yDirectionIncrement){

	if(zoom_.zoom == 1) return;

	int moveX = xSize_/2 + xDirectionIncrement;
	int moveY = ySize_/2 + yDirectionIncrement;

	computeZoom_(zoom_.zoom, moveX, moveY);	
}

template<TF::Size dim>
float TFWorkCopy<dim>::getZoom() const{

	return zoom_.zoom;
}

template<TF::Size dim>
float TFWorkCopy<dim>::getMaxZoom() const{

	return zoom_.max;
}

template<TF::Size dim>
void TFWorkCopy<dim>::setMaxZoom(const float zoom){
	
	zoom_.max = zoom;
}

template<TF::Size dim>
TF::Point<float, float> TFWorkCopy<dim>::getZoomCenter() const{

	return zoom_.center;
}

template<TF::Size dim>
void TFWorkCopy<dim>::computeZoom_(const float nextZoom, const int zoomX, const int zoomY){

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
	
	for(TF::Size i = 0; i < dim; ++i) changes_[i].setAllChanged();
}

//---update---

template<TF::Size dim>
typename TFAbstractFunction<dim>::Ptr TFWorkCopy<dim>::getFunctionMemento() const{

	return data_->clone();
}

template<TF::Size dim>
typename TFAbstractFunction<dim>::Ptr TFWorkCopy<dim>::getFunction() const{

	return data_;
}

template<TF::Size dim>
void TFWorkCopy<dim>::setDomain(const TF::Size domain){
	
	if(domain == data_->getDomain()) return;
	
	data_->resize(domain);

	if(histogram_ && histogram_->size() != data_->getDomain())
	{
		histogram_ = TF::Histogram::Ptr();
	}

	computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
}

template<TF::Size dim>
void TFWorkCopy<dim>::update(const typename TFAbstractFunction<dim>::Ptr function){

	data_ = function->clone();
	for(TF::Size i = 0; i < dim; ++i) changes_[i].setAllChanged();

	if(histogram_ && histogram_->size() != data_->getDomain())
	{
		data_->resize(histogram_->size());
	}

	computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
}
*/
} // namespace GUI
} // namespace M4D