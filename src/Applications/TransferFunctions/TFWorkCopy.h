#ifndef TF_WORKCOPY
#define TF_WORKCOPY

#include <TFTypes.h>
#include <TFAbstractFunction.h>

namespace M4D {
namespace GUI {

class TFWorkCopy{

public:

	typedef boost::shared_ptr<TFWorkCopy> Ptr;

	TFWorkCopy(const TFSize domain);
	~TFWorkCopy(){}

	TFColor getColor(const TFSize index);
	TFSize getViewSize();

	bool component1Changed();
	bool component2Changed();
	bool component3Changed();
	bool alphaChanged();
	bool histogramChanged();

	float getComponent1(const TFSize index);
	float getComponent2(const TFSize index);
	float getComponent3(const TFSize index);
	float getAlpha(const TFSize index);
	float getHistogramValue(const TFSize index);

	void setComponent1(const TFSize index, const float value);
	void setComponent2(const TFSize index, const float value);
	void setComponent3(const TFSize index, const float value);
	void setAlpha(const TFSize index, const float value);

	void zoomIn(const TFSize stepCount, const int zoomX, const int zoomY);
	void zoomOut(const TFSize stepCount, const int zoomX, const int zoomY);
	void zoom(const float zoom, const int zoomX, const float zoomY);
	void move(int xDirectionIncrement, int yDirectionIncrement);

	void setHistogramEnabled(bool value);
	bool histogramEnabled();

	float getZoom() const;
	float getMaxZoom() const;
	void setMaxZoom(const float zoom);
	TFPoint<float, float> getZoomCenter() const;

	void resize(const TFSize xSize, const TFSize ySize);

	void updateFunction(TFAbstractFunction::Ptr function);
	void update(TFAbstractFunction::Ptr function);

	void setHistogram(TFHistogramPtr histogram);
	
private:

	struct ZoomProperties{
		float zoom;
		float max;
		TFSize xOffset;
		float yOffset;
		float xRatio;
		int ratio;
		TFPoint<float,float> center;

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

	TFColorMapPtr data_;
	TFHistogramPtr histogram_;

	TFSize domain_;
	TFSize xSize_, ySize_;
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