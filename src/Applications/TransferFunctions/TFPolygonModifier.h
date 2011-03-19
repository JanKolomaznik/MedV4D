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

	enum Mode{
		Grayscale,
		RGB,
		HSV
	};

	TFPolygonModifier(TFWorkCopy::Ptr workCopy, Mode mode, bool alpha);
	~TFPolygonModifier();

	void mousePress(const int x, const int y, Qt::MouseButton button);
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

	Mode mode_;
	bool alpha_;

	bool leftMousePressed_;
	TF::PaintingPoint inputHelper_;

	bool zoomMovement_;
	TF::PaintingPoint zoomMoveHelper_;

	TF::Size baseRadius_;
	TF::Size topRadius_;

	void addPolygon_(const TF::PaintingPoint point);
	void addPoint_(const int x, const int y);

	void updateZoomTools_();
};

} // namespace GUI
} // namespace M4D

#endif //TF_POLYGON_MODIFIER