#ifndef TF_WORKCOPY
#define TF_WORKCOPY

#include <TFCommon.h>
#include <TFAbstractFunction.h>
#include <TFHistogram.h>

namespace M4D {
namespace GUI {

class TFWorkCopy{

public:

	typedef boost::shared_ptr<TFWorkCopy> Ptr;

	TFWorkCopy(TFAbstractFunction::Ptr function);
	~TFWorkCopy(){}

	TF::Color getColor(const TF::Size index);
	TF::Size getViewSize();

	bool component1Changed();
	bool component2Changed();
	bool component3Changed();
	bool alphaChanged();
	bool histogramChanged();

	float getComponent1(const TF::Size index);
	float getComponent2(const TF::Size index);
	float getComponent3(const TF::Size index);
	float getAlpha(const TF::Size index);
	float getHistogramValue(const TF::Size index);

	void setComponent1(const TF::Size index, const float value);
	void setComponent2(const TF::Size index, const float value);
	void setComponent3(const TF::Size index, const float value);
	void setAlpha(const TF::Size index, const float value);

	void zoomIn(const TF::Size stepCount, const int zoomX, const int zoomY);
	void zoomOut(const TF::Size stepCount, const int zoomX, const int zoomY);
	//void zoom(const float zoom, const int zoomX, const float zoomY);
	void move(int xDirectionIncrement, int yDirectionIncrement);

	void setHistogramEnabled(bool value);
	bool histogramEnabled();

	float getZoom() const;
	float getMaxZoom() const;
	void setMaxZoom(const float zoom);
	TF::Point<float, float> getZoomCenter() const;

	void resize(const TF::Size xSize, const TF::Size ySize);

	TFAbstractFunction::Ptr getFunctionMemento() const;
	TFAbstractFunction::Ptr getFunction() const;
	void update(const TFAbstractFunction::Ptr function);

	//void save(){data_->save()}
	//void load(){data_->load()}

	void setHistogram(const TF::Histogram::Ptr histogram);
	void setDomain(const TF::Size domain);
	
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

	TFAbstractFunction::Ptr data_;
	TF::Histogram::Ptr histogram_;

	TF::Size xSize_, ySize_;
	ZoomProperties zoom_;

	bool component1Changed_;
	bool component2Changed_;
	bool component3Changed_;
	bool alphaChanged_;
	bool histogramChanged_;

	bool histogramEnabled_;

	void computeZoom_(const float nextZoom, const int zoomX, const int zoomY);
};

} // namespace GUI
} // namespace M4D

#endif //TF_WORKCOPY