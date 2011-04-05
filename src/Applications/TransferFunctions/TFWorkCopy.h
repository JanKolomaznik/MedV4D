#ifndef TF_WORKCOPY
#define TF_WORKCOPY

#include <TFCommon.h>
#include <TFAbstractFunction.h>
#include <TFHistogram.h>

#include <cmath>

namespace M4D {
namespace GUI {

template<TF::Size dim>
class TFWorkCopy{

public:

	typedef typename boost::shared_ptr<TFWorkCopy<dim>> Ptr;

	enum ZoomDirection{
		ZoomX,
		ZoomY,
		ZoomBoth,
		ZoomNone
	};
	
	TFWorkCopy(typename TFAbstractFunction<dim>::Ptr function):
		data_(function),
		xSize_(0),
		ySize_(0),
		histogramChanged_(true),
		histogramEnabled_(false){
	}
	~TFWorkCopy(){}

	void save(TFXmlWriter::Ptr writer){

		writer->writeStartElement("WorkCopy");
				
			writer->writeAttribute("MaxZoom", TF::convert<float, std::string>(zoom_.max));
			writer->writeAttribute("ZoomX", TF::convert<float, std::string>(zoom_.xZoom));
			writer->writeAttribute("ZoomY", TF::convert<float, std::string>(zoom_.yZoom));
			writer->writeAttribute("X", TF::convert<float, std::string>(zoom_.center.x));
			writer->writeAttribute("Y", TF::convert<float, std::string>(zoom_.center.y));
			writer->writeAttribute("HistLogBase", TF::convert<long double, std::string>(hist_.logBase()));

		writer->writeEndElement();
	}
	bool load(TFXmlReader::Ptr reader){

		#ifndef TF_NDEBUG
			std::cout << "Loading work copy..." << std::endl;
		#endif

		bool ok = true;

		if(reader->readElement("WorkCopy"))
		{				
			float maxZoom = TF::convert<std::string, float>(reader->readAttribute("MaxZoom"));
			float zoomX = TF::convert<std::string, float>(reader->readAttribute("ZoomX"));
			float zoomY = TF::convert<std::string, float>(reader->readAttribute("ZoomY"));
			float x = TF::convert<std::string, float>(reader->readAttribute("X"));
			float y = TF::convert<std::string, float>(reader->readAttribute("Y"));
			long double logBase = TF::convert<std::string, long double>(reader->readAttribute("HistLogBase"));

			if(maxZoom < 1.0f)
			{
				maxZoom = zoom_.max;
				ok = false;
			}
			zoom_.max = maxZoom;

			if(zoomX < 1.0f || zoomX > maxZoom)
			{
				zoomX = zoom_.xZoom;
				ok = false;
			}
			if(zoomY < 1.0f || zoomY > maxZoom)
			{
				zoomY = zoom_.yZoom;
				ok = false;
			}
			if(x < 0.0f || x > 1.0f)
			{
				x = zoom_.center.x;
				ok = false;
			}
			if(y < 0.0f || y > 1.0f)
			{
				y = zoom_.center.y;
				ok = false;
			}
			computeZoomX_(zoomX, x, y);
			computeZoomY_(zoomY, x, y);

			if(logBase <= 1.0f)
			{
				ok = false;
			}
			else hist_.setLogBase(logBase);
		}
		else ok = false;

		return ok;
	}

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

	void increaseHistogramLogBase(const long double increment = 2.0){

		if(histogram_)
		{
			hist_.setLogBase(hist_.logBase()*increment);
			histogramChanged_ = true;
		}
	}
	void decreaseHistogramLogBase(const long double increment = 2.0){

		if(histogram_)
		{
			hist_.setLogBase(hist_.logBase()/increment);
			histogramChanged_ = true;
		}
	}

	//---getters---
	
	TF::Color getColor(const int index, const TF::Size dimension){

		tfAssert(index >= 0 && index < (int)xSize_);

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		TF::Color result(0,0,0,0);
		int count = 0;
		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				result += data_->getMappedRGBfColor(i, dimension);
				++count;
			}
		}
		result /= count;
		result.alpha = 1;
		tfAssert(count > 0);
		return result;
	}
	
	float getComponent1(const int index, const TF::Size dimension){

		tfAssert(index >= 0 && index < (int)xSize_);

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		float result = 0;
		int count = 0;
		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				result += ((*data_)[i][dimension].component1 - zoom_.yOffset)*zoom_.yZoom;
				++count;
			}
		}
		tfAssert(count > 0);
		return result/count;
	}
	
	float getComponent2(const int index, const TF::Size dimension){

		tfAssert(index >= 0 && index < (int)xSize_);

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		float result = 0;
		int count = 0;
		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				result += ((*data_)[i][dimension].component2 - zoom_.yOffset)*zoom_.yZoom;
				++count;
			}
		}
		tfAssert(count > 0);
		return result/count;
	}
	
	float getComponent3(const int index, const TF::Size dimension){

		tfAssert(index >= 0 && index < (int)xSize_);

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		float result = 0;
		int count = 0;
		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				result += ((*data_)[i][dimension].component3 - zoom_.yOffset)*zoom_.yZoom;
				++count;
			}
		}
		tfAssert(count > 0);
		return result/count;
	}
	
	float getAlpha(const int index, const TF::Size dimension){

		tfAssert(index >= 0 && index < (int)xSize_);

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		float result = 0;
		int count = 0;
		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				result += ((*data_)[i][dimension].alpha - zoom_.yOffset)*zoom_.yZoom;
				++count;
			}
		}
		tfAssert(count > 0);
		return result/count;
	}
	
	float getHistogramValue(const int index){

		tfAssert(index >= 0 && index < (int)xSize_);

		if(!histogramEnabled_ || !histogram_) return -1;

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		float result = 0;
		int count = 0;
		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				result += (hist_.getExpLogValue((*histogram_)[i]) - zoom_.yOffset)*zoom_.yZoom;
				++count;
			}
		}
		tfAssert(count > 0);
		return result/count;
	}

	//---setters---
	
	void setComponent1(const int index, const TF::Size dimension, const float value){

		float correctedValue = value/zoom_.yZoom + zoom_.yOffset;
		if(correctedValue < 0) correctedValue = 0;
		if(correctedValue > 1) correctedValue = 1;

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				(*data_)[i][dimension].component1 = correctedValue;
			}
		}
		changes_[dimension - 1].component1 = true;
	}
	
	void setComponent2(const int index, const TF::Size dimension, const float value){

		float correctedValue = value/zoom_.yZoom + zoom_.yOffset;
		if(correctedValue < 0) correctedValue = 0;
		if(correctedValue > 1) correctedValue = 1;

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				(*data_)[i][dimension].component2 = correctedValue;
			}
		}
		changes_[dimension - 1].component2 = true;
	}
	
	void setComponent3(const int index, const TF::Size dimension, const float value){

		float correctedValue = value/zoom_.yZoom + zoom_.yOffset;
		if(correctedValue < 0) correctedValue = 0;
		if(correctedValue > 1) correctedValue = 1;

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				(*data_)[i][dimension].component3 = correctedValue;
			}
		}
		changes_[dimension - 1].component3 = true;
	}
	
	void setAlpha(const int index, const TF::Size dimension, const float value){

		float correctedValue = value/zoom_.yZoom + zoom_.yOffset;
		if(correctedValue < 0) correctedValue = 0;
		if(correctedValue > 1) correctedValue = 1;

		float xRatio = data_->getDomain()/(xSize_*zoom_.xZoom);
		float indexBase = index*xRatio + zoom_.xOffset;
		float radius = xRatio/2.0;
		float bottom = indexBase - radius;
		float top = indexBase + radius;

		for(int i = bottom; i < top; ++i)
		{
			if(i >= 0 && i < (int)data_->getDomain())
			{
				(*data_)[i][dimension].alpha = correctedValue;
			}
		}
		changes_[dimension - 1].alpha = true;
	}

	//---size---
	
	void resize(const TF::Size xSize, const TF::Size ySize){

		xSize_ = xSize;
		ySize_ = ySize;	
		computeZoomX_(zoom_.xZoom, 0.5f, 0.5f);
		computeZoomY_(zoom_.yZoom, 0.5f, 0.5f);
	}

	//---zoom---
	
	void zoomIn(const TF::Size stepCount, const int centerX, const int centerY,
		const ZoomDirection direction = ZoomX){

		if(direction == ZoomNone) return;

		float nextZoom;
		float x = centerX/(float)xSize_;
		float y = centerY/(float)ySize_;
		if(direction != ZoomY)
		{
			if(zoom_.xZoom == zoom_.max) return;

			nextZoom = zoom_.xZoom + stepCount;
			if(nextZoom > zoom_.max) nextZoom = zoom_.max;
			if(nextZoom < 1) nextZoom = 1;
			if(nextZoom == zoom_.xZoom) return;

			computeZoomX_(nextZoom, x, y);
		}
		if(direction != ZoomX)
		{
			if(zoom_.yZoom == zoom_.max) return;

			nextZoom = zoom_.yZoom + stepCount;
			if(nextZoom > zoom_.max) nextZoom = zoom_.max;
			if(nextZoom < 1) nextZoom = 1;
			if(nextZoom == zoom_.yZoom) return;

			computeZoomY_(nextZoom, x, y);
		}
	}
	
	void zoomOut(const TF::Size stepCount, const int centerX, const int centerY,
		const ZoomDirection direction = ZoomX){

		if(direction == ZoomNone) return;

		float nextZoom;
		float x = centerX/(float)xSize_;
		float y = centerY/(float)ySize_;
		if(direction != ZoomY)
		{
			nextZoom = zoom_.xZoom - stepCount;
			if(nextZoom > zoom_.max) nextZoom = zoom_.max;
			if(nextZoom < 1) nextZoom = 1;
			if(nextZoom != zoom_.xZoom) computeZoomX_(nextZoom, x, y);
		}
		if(direction != ZoomX)
		{
			nextZoom = zoom_.yZoom - stepCount;
			if(nextZoom > zoom_.max) nextZoom = zoom_.max;
			if(nextZoom < 1) nextZoom = 1;
			if(nextZoom != zoom_.yZoom) computeZoomY_(nextZoom, x, y);
		}
	}
	
	void move(int xDirectionIncrement, int yDirectionIncrement){

		if(zoom_.xZoom == 1 && zoom_.yZoom == 1) return;

		float x = (xSize_/2.0f + xDirectionIncrement)/(float)xSize_;
		float y = (ySize_/2.0f + yDirectionIncrement)/(float)ySize_;

		computeZoomX_(zoom_.xZoom, x, y);	
		computeZoomY_(zoom_.yZoom, x, y);	
	}
	
	float getZoomX() const{

		return zoom_.xZoom;
	}
	
	float getZoomY() const{

		return zoom_.yZoom;
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

		computeZoomX_(zoom_.xZoom, 0.5f, 0.5f);
		computeZoomY_(zoom_.yZoom, 0.5f, 0.5f);
	}
	
	void update(const typename TFAbstractFunction<dim>::Ptr function){

		data_ = function->clone();

		if(histogram_ && histogram_->size() != data_->getDomain())
		{
			data_->resize(histogram_->size());
		}

		computeZoomX_(zoom_.xZoom, 0.5f, 0.5f);
		computeZoomY_(zoom_.yZoom, 0.5f, 0.5f);
	}
	
private:

	class HistProperties{
	public:
		HistProperties():
			logBase_(65536.0),
			logMod_(std::log(logBase_)){
		}

		void setLogBase(const long double logBase){

			if(logBase <= 1.0 || logBase == HUGE_VAL || logBase == logBase_) return;
			logBase_ = logBase;
			logMod_ = std::log(logBase_);
		}
		long double logBase(){

			return logBase_;
		}

		double getLogValue(const TF::Size value){

			if(value == 0) return 0.0;
			
			return std::log((double)value)/logMod_;			
		}
		double getExpLogValue(const TF::Size value){

			return 1.0 - std::exp(-getLogValue(value));
		}
	private:
		long double logBase_;
		double logMod_;
	};

	struct ZoomProperties{
		float xOffset;
		float xZoom;
		float yOffset;
		float yZoom;
		float max;
		TF::Point<float,float> center;

		ZoomProperties():
			xOffset(0.0f),
			xZoom(1.0f),
			yOffset(0.0f),
			yZoom(1.0f),
			max(40.0f),
			center(0.5f,0.5f){
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
	ZoomProperties zoom_;

	DimensionChange changes_[dim];

	bool histogramChanged_;
	bool histogramEnabled_;
	HistProperties hist_;
	
	void computeZoomX_(const float nextZoom, const float centerX, const float centerY){

		float relativeZoomedRatioX = data_->getDomain()/zoom_.xZoom;

		float xRadius = (data_->getDomain()/nextZoom)/2.0f;
		float xOffesetInc = zoom_.xOffset + centerX*relativeZoomedRatioX - xRadius;

		float maxXOffset = data_->getDomain() - 2.0f*xRadius;		
		if(xOffesetInc < 0.0f) xOffesetInc = 0.0f;
		if(xOffesetInc > maxXOffset) xOffesetInc = maxXOffset;

		zoom_.xOffset = xOffesetInc;
		zoom_.xZoom = nextZoom;

		zoom_.center.x =(((data_->getDomain()/zoom_.xZoom)/2.0f) + zoom_.xOffset)/data_->getDomain();
		
		for(TF::Size i = 0; i < dim; ++i) changes_[i].setAllChanged();
		histogramChanged_ = true;
	}
	
	void computeZoomY_(const float nextZoom, const float centerX, const float centerY){

		float relativeZoomedRatioY = 1.0f/zoom_.yZoom;

		float yRadius = (1.0f/nextZoom)/2.0f;
		float yOffesetInc = zoom_.yOffset + centerY*relativeZoomedRatioY - yRadius;

		float maxYOffset = 1 - 2.0f*yRadius;
		if(yOffesetInc < 0.0f) yOffesetInc = 0.0f;
		if(yOffesetInc > maxYOffset) yOffesetInc = maxYOffset;

		zoom_.yOffset = yOffesetInc;
		zoom_.yZoom = nextZoom;

		zoom_.center.y = (1.0f/zoom_.yZoom)/2.0f + zoom_.yOffset;
		
		for(TF::Size i = 0; i < dim; ++i) changes_[i].setAllChanged();
		histogramChanged_ = true;
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_WORKCOPY