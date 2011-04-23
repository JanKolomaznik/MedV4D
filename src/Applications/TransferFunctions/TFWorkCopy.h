#ifndef TF_WORKCOPY
#define TF_WORKCOPY

#include <TFCommon.h>
#include <TFFunctionInterface.h>
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
	
	TFFunctionInterface::Ptr getFunction();
	
	TF::Size getDimension();
	void setDataStructure(const std::vector<TF::Size>& dataStructure);

	//---change---
	
	bool changed();
	bool histogramChanged();

	void forceUpdate(const bool updateHistogram = false);

	//---histogram---
	
	void setHistogram(const TF::Histogram::Ptr histogram);	
	void setHistogramEnabled(bool value);
	bool histogramEnabled();
	void increaseHistogramLogBase(const long double increment = 1.0);
	void decreaseHistogramLogBase(const long double increment = 1.0);

	//---getters---
	
	TF::Color getRGBfColor(const TF::Coordinates& coords);
	TF::Color getColor(const TF::Coordinates& coords);
	float getHistogramValue(const int index);

	//---setters---
	
	void setComponent1(const TF::Coordinates& coords, const float value);
	void setComponent2(const TF::Coordinates& coords, const float value);
	void setComponent3(const TF::Coordinates& coords, const float value);	
	void setAlpha(const TF::Coordinates& coords, const float value);
	void setColor(const TF::Coordinates& coords, const float value);

	//---size---
	
	void resize(const TF::Size dimension, const TF::Size size);
	void resize(const std::vector<TF::Size>& sizes);
	void resizeHistogram(const TF::Size size);

	//---zoom---
	
	void zoom(const TF::Size dimension, const int center, const int stepCount);
	void move(const std::vector<int>& increments);
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

	enum Component{
		Component1,
		Component2,
		Component3,
		Alpha
	};

	static const TF::Size histogramIndex = 0;

	TFFunctionInterface::Ptr data_;
	TF::Histogram::Ptr histogram_;

	TF::Coordinates coords_;

	std::vector<TF::Size> sizes_;

	ZoomProperties zoom_;

	bool changed_;

	bool histogramChanged_;
	bool histogramEnabled_;
	HistProperties hist_;
	
	TF::Color getColorFromZoomedArea_(const TF::Coordinates& coords,
		TF::Size& count,
		const bool& RGBf,
		TF::Size dimension = 1);
	
	void TFWorkCopy::setComponentToZoomedArea_(const TF::Coordinates& coords,
		const Component& component,
		const float& value,
		TF::Size dimension = 1);

	void computeZoom_(const TF::Size dimension, const float nextZoom, const float center);	
};

} // namespace GUI
} // namespace M4D

#endif //TF_WORKCOPY