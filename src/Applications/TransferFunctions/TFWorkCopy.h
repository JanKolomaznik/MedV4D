#ifndef TF_WORKCOPY
#define TF_WORKCOPY

#include <TFCommon.h>
#include <TFAbstractFunction.h>
#include <TFHistogram.h>

#include <cmath>

namespace M4D {
namespace GUI {

class TFWorkCopy{

public:

	typedef boost::shared_ptr<TFWorkCopy> Ptr;

	static const TF::Size noZoom = -1;
	
	TFWorkCopy(TFFunctionInterface::Ptr function);
	~TFWorkCopy();

	void save(TF::XmlWriterInterface* writer);
	bool load(TF::XmlReaderInterface* reader, bool& sideError);
	
	//TFFunctionInterface::Const getFunctionMemento();	
	TFFunctionInterface::Ptr getFunction();
	
	TF::Size getDimension();
	void setDataStructure(const std::vector<TF::Size>& dataStructure);

	//void update(TFFunctionInterface::Const function);

	//---change---
	
	bool component1Changed(const TF::Size dimension);
	bool component2Changed(const TF::Size dimension);
	bool component3Changed(const TF::Size dimension);
	bool alphaChanged(const TF::Size dimension);
	bool histogramChanged();

	void forceUpdate(const bool updateHistogram = false);

	//---histogram---
	
	void setHistogram(const TF::Histogram::Ptr histogram);	
	void setHistogramEnabled(bool value);
	bool histogramEnabled();
	void increaseHistogramLogBase(const long double increment = 1.0);
	void decreaseHistogramLogBase(const long double increment = 1.0);

	//---getters---
	
	TF::Color getColor(const TF::Size dimension, const int index);
	float getComponent1(const TF::Size dimension, const int index);
	float getComponent2(const TF::Size dimension, const int index);
	float getComponent3(const TF::Size dimension, const int index);
	float getAlpha(const TF::Size dimension, const int index);
	float getHistogramValue(const int index);

	//---setters---
	
	void setComponent1(const TF::Size dimension, const int index, const float value);
	void setComponent2(const TF::Size dimension, const int index, const float value);
	void setComponent3(const TF::Size dimension, const int index, const float value);	
	void setAlpha(const TF::Size dimension, const int index, const float value);

	//---size---
	
	void resize(const TF::Size dimension, const TF::Size size);
	void resizeHistogram(const TF::Size size);

	//---zoom---
	
	void zoom(const TF::Size dimension, const int center, const int stepCount);
	void move(const std::vector<int> increments);
	void zoomHistogram(const int center, const int stepCount);
	void moveHistogram(const int increment);
	
	float getZoom(const TF::Size dimension);
	float getZoomCenter(const TF::Size dimension);	
	float getMaxZoom();	
	void setMaxZoom(const float zoom);
	
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
		std::vector<float> zoom;
		std::vector<float> center;
		std::vector<float> offset;
		float max;

		ZoomProperties(TF::Size dimension):
			zoom(dimension+1, 1.0f),
			center(dimension+1, 0.5f),
			offset(dimension+1, 0.0f),
			max(40.0f){
		}

		void reset(){
			TF::Size size = zoom.size();
			zoom = std::vector<float>(size, 1.0f);
			center = std::vector<float>(size, 0.5f);
			offset = std::vector<float>(size, 0.0f);
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

	static const TF::Size histogramIndex = 0;

	TFFunctionInterface::Ptr data_;
	TF::Histogram::Ptr histogram_;

	std::vector<TF::Size> sizes_;
	ZoomProperties zoom_;

	std::vector<DimensionChange> changes_;

	bool histogramChanged_;
	bool histogramEnabled_;
	HistProperties hist_;
	
	void computeZoom_(const TF::Size dimension, const float nextZoom, const float center);	
};

} // namespace GUI
} // namespace M4D

#endif //TF_WORKCOPY