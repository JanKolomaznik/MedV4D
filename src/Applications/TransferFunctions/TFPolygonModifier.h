#ifndef TF_POLYGON_MODIFIER
#define TF_POLYGON_MODIFIER

#include <TFAbstractModifier.h>
#include <ui_TFPolygonModifier.h>

namespace M4D {
namespace GUI {

class TFPolygonModifier: public TFAbstractModifier{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFPolygonModifier> Ptr;

	TFPolygonModifier(TFAbstractModifier::Type type, const TFSize domain);
	~TFPolygonModifier();

	void mousePress(const int x, const int y, MouseButton button);
	void mouseRelease(const int x, const int y);
	void mouseMove(const int x, const int y);
	void mouseWheel(const int steps, const int x, const int y);

private slots:

	void activeView_changed(int index);
	void histogram_check(bool enabled);

	void bottomSpin_changed(int value);
	void topSpin_changed(int value);

	void maxZoomSpin_changed(int value);

private:

	enum ActiveView{
		Active1,
		Active2,
		Active3,
		ActiveAlpha
	};	
	ActiveView activeView_;
	
	Ui::TFPolygonModifier* tools_;

	void active1Next_();
	void active2Next_();
	void active3Next_();
	void activeAlphaNext_();

	TFAbstractModifier::Type type_;

	bool leftMousePressed_;
	TFPaintingPoint inputHelper_;

	bool zoomMovement_;
	TFPaintingPoint zoomMoveHelper_;

	TFSize baseRadius_;
	TFSize topRadius_;

	void addPolygon_(const TFPaintingPoint point);
	void addPoint_(const int x, const int y);

	void updateZoomTools_();
};

} // namespace GUI
} // namespace M4D

#endif //TF_POLYGON_MODIFIER