#ifndef TF_WORKCOPY
#define TF_WORKCOPY

#include <TFTypes.h>
#include <TFAbstractFunction.h>

namespace M4D {
namespace GUI {

class TFWorkCopy{

public:

	struct ZoomProperties{
		float zoom;
		float maxZoom;
		TFSize xOffset;
		float yOffset;

		ZoomProperties():
			zoom(1),
			maxZoom(20),
			xOffset(0),
			yOffset(0){}

		void reset(){
			zoom = 1;
			xOffset = 0;
			yOffset = 0;
		}
	};

	typedef boost::shared_ptr<TFWorkCopy> Ptr;

	TFWorkCopy(const TFSize& domain);
	~TFWorkCopy(){}

	TFColor getColor(const TFSize& index);

	float getComponent1(const TFSize& index);
	float getComponent2(const TFSize& index);
	float getComponent3(const TFSize& index);
	float getAlpha(const TFSize& index);

	void setComponent1(const TFSize& index, const float& value);
	void setComponent2(const TFSize& index, const float& value);
	void setComponent3(const TFSize& index, const float& value);
	void setAlpha(const TFSize& index, const float& value);

	void zoomIn(const TFSize& stepCount, const TFSize& inputX, const TFSize& inputY);
	void zoomOut(const TFSize& stepCount, const TFSize& inputX, const TFSize& inputY);
	void move(int xDirectionIncrement, int yDirectionIncrement);

	const float& zoom();

	TFSize size();
	void resize(const TFSize& xSize, const TFSize& ySize);

	void updateFunction(TFAbstractFunction::Ptr function);
	void update(TFAbstractFunction::Ptr function);
	
private:

	TFColorMapPtr data_;
	TFSize domain_;
	ZoomProperties zoom_;
	TFSize xSize_, ySize_;

	void computeZoom_(const float& nextZoom, const TFSize& inputX, const TFSize& inputY);

};

} // namespace GUI
} // namespace M4D

#endif //TF_WORKCOPY