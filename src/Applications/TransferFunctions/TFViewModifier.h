#ifndef TF_VIEW_MODIFIER
#define TF_VIEW_MODIFIER

#include <TFAbstractModifier.h>

#include <QtGui/QGridLayout>
#include <QtGui/QSpacerItem>

#include <ui_TFViewModifier.h>

namespace M4D {
namespace GUI {

#define TF_DIMENSION_1 1

class TFViewModifier: public TFAbstractModifier<TF_DIMENSION_1>{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFViewModifier> Ptr;

	typedef TFWorkCopy<TF_DIMENSION_1> WorkCopy;

	TFViewModifier(WorkCopy::Ptr workCopy);
	~TFViewModifier();

	virtual bool load(TFXmlReader::Ptr reader);

protected slots:

	void histogram_check(bool enabled);

	void maxZoomSpin_changed(int value);
	void xAxis_check(bool enabled);
	void yAxis_check(bool enabled);

protected:

	Ui::TFViewModifier* viewTools_;
	QWidget* viewWidget_;

	bool altPressed_;

	bool zoomMovement_;
	TF::PaintingPoint zoomMoveHelper_;
	WorkCopy::ZoomDirection zoomDirection_;

	virtual void createTools_();

	virtual void mousePressEvent(QMouseEvent *e);
	virtual void mouseReleaseEvent(QMouseEvent *e);
	virtual void mouseMoveEvent(QMouseEvent *e);
	virtual void wheelEvent(QWheelEvent *e);

	virtual void keyPressEvent(QKeyEvent *e);
	virtual void keyReleaseEvent(QKeyEvent *e);

	virtual void addPoint_(const int x, const int y){}

	void updateZoomTools_();
	
	QGridLayout* centerWidget_(QWidget *widget);
};

} // namespace GUI
} // namespace M4D

#endif //TF_VIEW_MODIFIER