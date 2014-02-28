#ifndef TF_POLYGON_MODIFIER
#define TF_POLYGON_MODIFIER

#include "MedV4D/GUI/TF/Modifier1D.h"

#include "MedV4D/generated/ui_PolygonModifier.h"

namespace M4D {
namespace GUI {

class PolygonModifier: public Modifier1D{

	Q_OBJECT

public:

	typedef boost::shared_ptr<PolygonModifier> Ptr;

	PolygonModifier(
		FunctionInterface::Ptr function,
		Painter1D::Ptr painter);

	~PolygonModifier();

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
	
	Ui::PolygonModifier* polygonTools_;
	QWidget* polygonWidget_;

	std::vector<ScrollMode> scrollModes_;

	const TF::Size polygonSpinStep_;

	TF::Size baseRadius_;
	TF::Size topRadius_;

	void createTools_();

	void addPolygon_(const TF::PaintingPoint point);
};

} // namespace GUI
} // namespace M4D

#endif //TF_POLYGON_MODIFIER
