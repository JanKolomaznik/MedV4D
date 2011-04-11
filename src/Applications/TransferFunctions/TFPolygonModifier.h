#ifndef TF_POLYGON_MODIFIER
#define TF_POLYGON_MODIFIER

#include <TFSimpleModifier.h>

#include <ui_TFPolygonModifier.h>

namespace M4D {
namespace GUI {

class TFPolygonModifier: public TFSimpleModifier{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFPolygonModifier> Ptr;

	typedef TFSimpleModifier::WorkCopy WorkCopy;

	typedef TFSimpleModifier::Mode Mode;

	TFPolygonModifier(WorkCopy::Ptr workCopy, Mode mode, bool alpha);
	~TFPolygonModifier();

protected:
	
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);
	void wheelEvent(QWheelEvent *e);

	void keyPressEvent(QKeyEvent *e);
	void keyReleaseEvent(QKeyEvent *e);

private slots:

	void baseSpin_changed(int value);
	void topSpin_changed(int value);

private:

	enum ScrollMode{
		ScrollZoom,
		ScrollHistogram,
		ScrollBase,
		ScrollTop
	};
	
	Ui::TFPolygonModifier* polygonTools_;
	QWidget* polygonWidget_;

	std::vector<ScrollMode> scrollModes_;

	const TF::Size radiusStep_;

	TF::Size baseRadius_;
	TF::Size topRadius_;

	void createTools_();

	void addPolygon_(const TF::PaintingPoint point);
};

} // namespace GUI
} // namespace M4D

#endif //TF_POLYGON_MODIFIER