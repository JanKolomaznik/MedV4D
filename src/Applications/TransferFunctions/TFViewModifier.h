#ifndef TF_VIEW_MODIFIER
#define TF_VIEW_MODIFIER

#include <TFAbstractModifier.h>

#include <QtGui/QGridLayout>
#include <QtGui/QSpacerItem>

#include <ui_TFViewModifier.h>
#include <ui_TFDimensionZoom.h>

namespace M4D {
namespace GUI {

class TFViewModifier: public TFAbstractModifier{

	Q_OBJECT

public:

	typedef boost::shared_ptr<TFViewModifier> Ptr;

	~TFViewModifier();

	void setDataStructure(const std::vector<TF::Size>& dataStructure);

protected slots:

	void histogram_check(bool enabled);
	void maxZoomSpin_changed(int value);

protected:

	TFViewModifier(TFFunctionInterface::Ptr function, TFAbstractPainter::Ptr painter);

	Ui::TFViewModifier* viewTools_;
	std::vector<Ui::TFDimensionZoom*> dimensionsUi_;
	QWidget* viewWidget_;

	bool altPressed_;

	bool zoomMovement_;
	TF::PaintingPoint zoomMoveHelper_;

	virtual void createTools_();
	
	virtual void computeInput_() = 0;
	virtual std::vector<int> computeZoomMoveIncrements_(const int moveX, const int moveY) = 0;

	virtual bool loadSettings_(TF::XmlReaderInterface* reader);

	virtual void mousePressEvent(QMouseEvent *e);
	virtual void mouseReleaseEvent(QMouseEvent *e);
	virtual void mouseMoveEvent(QMouseEvent *e);
	virtual void wheelEvent(QWheelEvent *e);

	virtual void keyPressEvent(QKeyEvent *e);
	virtual void keyReleaseEvent(QKeyEvent *e);

	TF::PaintingPoint getRelativePoint_(const int x, const int y, bool acceptOutOfBounds = false);

	void updateZoomTools_();
	
	QGridLayout* centerWidget_(QWidget *widget,
		bool top = true, bool bottom = true,
		bool left = true, bool right = true);
};

} // namespace GUI
} // namespace M4D

#endif //TF_VIEW_MODIFIER