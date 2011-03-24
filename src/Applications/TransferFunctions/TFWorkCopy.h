#ifndef TF_WORKCOPY
#define TF_WORKCOPY

#include <TFCommon.h>
#include <TFAbstractFunction.h>
#include <TFHistogram.h>

namespace M4D {
namespace GUI {

template<TF::Size dim>
class TFWorkCopy{

public:

	typedef typename boost::shared_ptr<TFWorkCopy<dim>> Ptr;
	
	TFWorkCopy(typename TFAbstractFunction<dim>::Ptr function):
		data_(function),
		xSize_(0),
		ySize_(0),
		histogramChanged_(true),
		histogramEnabled_(false){
	}

	~TFWorkCopy(){}

	//---change---
	
	bool component1Changed(const TF::Size dimension){

		if(changes_[dimension - 1].component1)
		{
			changes_[dimension - 1].component1 = false;
			return true;
		}
		return false;
	}
	
	bool component2Changed(const TF::Size dimension){

		if(changes_[dimension - 1].component2)
		{
			changes_[dimension - 1].component2 = false;
			return true;
		}
		return false;
	}
	
	bool component3Changed(const TF::Size dimension){

		if(changes_[dimension - 1].component3)
		{
			changes_[dimension - 1].component3 = false;
			return true;
		}
		return false;
	}
	
	bool alphaChanged(const TF::Size dimension){

		if(changes_[dimension - 1].alpha)
		{
			changes_[dimension - 1].alpha = false;
			return true;
		}
		return false;
	}
	
	bool histogramChanged(){

		if(histogramChanged_)
		{
			histogramChanged_ = false;
			return true;
		}
		return false;
	}

	//---histogram---
	
	void setHistogram(const TF::Histogram::Ptr histogram){

		histogram_ = histogram;
		setDomain(histogram->size());
		histogramChanged_ = true;
	}
	
	void setHistogramEnabled(bool value){

		if(histogramEnabled_ != value) histogramChanged_ = true;
		histogramEnabled_ = value;
	}
	
	bool histogramEnabled(){

		return histogramEnabled_;
	}

	//---getters---
	
	TF::Color getColor(const TF::Size index, const TF::Size dimension){

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
				result += data_->getMappedRGBfColor(currIndex, dimension);
				++count;
			}
		}
		result /= count;
		result.alpha = 1;
		tfAssert(count > 0);
		return result;
	}
	
	float getComponent1(const TF::Size index, const TF::Size dimension){

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
	
	float getComponent2(const TF::Size index, const TF::Size dimension){

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
	
	float getComponent3(const TF::Size index, const TF::Size dimension){

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
	
	float getAlpha(const TF::Size index, const TF::Size dimension){

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
	
	float getHistogramValue(const TF::Size index){

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
	
	void setComponent1(const TF::Size index, const TF::Size dimension, const float value){

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
	
	void setComponent2(const TF::Size index, const TF::Size dimension, const float value){

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
	
	void setComponent3(const TF::Size index, const TF::Size dimension, const float value){

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
	
	void setAlpha(const TF::Size index, const TF::Size dimension, const float value){

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
	
	void resize(const TF::Size xSize, const TF::Size ySize){

		xSize_ = xSize;
		ySize_ = ySize;	
		computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
	}

	//---zoom---
	
	void zoomIn(const TF::Size stepCount, const int zoomX, const int zoomY){

		if(zoom_.zoom == zoom_.max) return;

		float nextZoom = zoom_.zoom + stepCount;	
		if(nextZoom > zoom_.max) nextZoom = zoom_.max;

		computeZoom_(nextZoom, zoomX, zoomY);
	}
	
	void zoomOut(const TF::Size stepCount, const int zoomX, const int zoomY){

		if(zoom_.zoom == 1) return;

		float nextZoom = zoom_.zoom - stepCount;
		if(nextZoom > zoom_.max) nextZoom = zoom_.max;
		if(nextZoom < 1) nextZoom = 1;

		computeZoom_(nextZoom, zoomX, zoomY);
	}
	
	void move(int xDirectionIncrement, int yDirectionIncrement){

		if(zoom_.zoom == 1) return;

		int moveX = xSize_/2 + xDirectionIncrement;
		int moveY = ySize_/2 + yDirectionIncrement;

		computeZoom_(zoom_.zoom, moveX, moveY);	
	}
	
	float getZoom() const{

		return zoom_.zoom;
	}
	
	float getMaxZoom() const{

		return zoom_.max;
	}
	
	void setMaxZoom(const float zoom){
		
		zoom_.max = zoom;
	}
	
	TF::Point<float, float> getZoomCenter() const{

		return zoom_.center;
	}

	//---update---
	
	typename TFAbstractFunction<dim>::Ptr getFunctionMemento() const{

		return data_->clone();
	}
	
	typename TFAbstractFunction<dim>::Ptr getFunction() const{

		return data_;
	}
	
	void setDomain(const TF::Size domain){
		
		if(domain == data_->getDomain()) return;
		
		data_->resize(domain);

		if(histogram_ && histogram_->size() != data_->getDomain())
		{
			histogram_ = TF::Histogram::Ptr();
		}

		computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
	}
	
	void update(const typename TFAbstractFunction<dim>::Ptr function){

		data_ = function->clone();

		if(histogram_ && histogram_->size() != data_->getDomain())
		{
			data_->resize(histogram_->size());
		}

		computeZoom_(zoom_.zoom, xSize_/2, ySize_/2);
	}
	
private:

	struct ZoomProperties{
		float zoom;
		float max;
		TF::Size xOffset;
		float yOffset;
		float xRatio;
		int ratio;
		TF::Point<float,float> center;

		ZoomProperties():
			zoom(1),
			max(40),
			xOffset(0),
			yOffset(0),
			xRatio(1),
			ratio(1),
			center(0,0){
		}
	};

	struct DimensionChange{
		bool component1;
		bool component2;
		bool component3;
		bool alpha;

		DimensionChange():
			component1(true),
			component2(true),
			component3(true),
			alpha(true){
		}

		void setAllChanged(){
			component1 = true;
			component2 = true;
			component3 = true;
			alpha = true;
		}
	};

	typename TFAbstractFunction<dim>::Ptr data_;
	TF::Histogram::Ptr histogram_;

	TF::Size xSize_, ySize_;
	TF::Size dimension_;
	ZoomProperties zoom_;

	DimensionChange changes_[dim];
	bool histogramChanged_;

	bool histogramEnabled_;
	
	void computeZoom_(const float nextZoom, const int zoomX, const int zoomY){

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
		histogramChanged_ = true;
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_WORKCOPY