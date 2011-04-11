#include "TFPolygonModifier.h"

namespace M4D {
namespace GUI {

TFPolygonModifier::TFPolygonModifier(WorkCopy::Ptr workCopy,  Mode mode, bool alpha):
	TFSimpleModifier(workCopy, mode, alpha),
	polygonTools_(new Ui::TFPolygonModifier),
	polygonWidget_(new QWidget),
	baseRadius_(50),
	topRadius_(20),
	radiusStep_(5){

	polygonTools_->setupUi(polygonWidget_);

	bool topSpinConnected = QObject::connect(polygonTools_->topSpin, SIGNAL(valueChanged(int)),
		this, SLOT(topSpin_changed(int)));
	tfAssert(topSpinConnected);
	bool baseSpinConnected = QObject::connect(polygonTools_->baseSpin, SIGNAL(valueChanged(int)),
		this, SLOT(baseSpin_changed(int)));
	tfAssert(baseSpinConnected);

	scrollModes_.push_back(ScrollZoom);
}

TFPolygonModifier::~TFPolygonModifier(){}

void TFPolygonModifier::createTools_(){

    QFrame* vSeparator = new QFrame();
    vSeparator->setFrameShape(QFrame::HLine);
    vSeparator->setFrameShadow(QFrame::Sunken);

	QVBoxLayout* vLayout = new QVBoxLayout();
	vLayout->addItem(centerWidget_(simpleWidget_));
	vLayout->addWidget(vSeparator);
	vLayout->addItem(centerWidget_(polygonWidget_));

    QFrame* hSeparator = new QFrame();
    hSeparator->setFrameShape(QFrame::VLine);
    hSeparator->setFrameShadow(QFrame::Sunken);

	QHBoxLayout* hLayout = new QHBoxLayout();
	hLayout->addItem(centerWidget_(viewWidget_));
	hLayout->addWidget(hSeparator);
	hLayout->addItem(vLayout);

	toolsWidget_ = new QWidget();
	toolsWidget_->setLayout(hLayout);
}

void TFPolygonModifier::topSpin_changed(int value){

	topRadius_ = value;
	if(baseRadius_ < topRadius_) polygonTools_->baseSpin->setValue(topRadius_);
}

void TFPolygonModifier::baseSpin_changed(int value){

	baseRadius_ = value;
	if(baseRadius_ < topRadius_) polygonTools_->topSpin->setValue(baseRadius_);
}

void TFPolygonModifier::mouseReleaseEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_) addPolygon_(relativePoint);

	leftMousePressed_ = false;

	TFViewModifier::mouseReleaseEvent(e);
}

void TFPolygonModifier::mouseMoveEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_)
	{
		for(;inputHelper_.x < relativePoint.x; ++inputHelper_.x)
		{
			addPoint_(inputHelper_.x - baseRadius_, 0);
		}

		addPolygon_(relativePoint);

		for(;inputHelper_.x > relativePoint.x; --inputHelper_.x)
		{
			addPoint_(inputHelper_.x + baseRadius_, 0);
		}
	}

	TFViewModifier::mouseMoveEvent(e);
}

void TFPolygonModifier::wheelEvent(QWheelEvent *e){

	int steps = e->delta() / 120;
	if(steps == 0) return;

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y());
	if(relativePoint == ignorePoint_) return;

	switch(scrollModes_.back())
	{
		case ScrollBase:
		{
			polygonTools_->baseSpin->setValue(polygonTools_->baseSpin->value() + radiusStep_*steps);	
			break;
		}
		case ScrollTop:
		{
			polygonTools_->topSpin->setValue(polygonTools_->topSpin->value() + radiusStep_*steps);	
			break;
		}
		default:
		{			
			TFViewModifier::wheelEvent(e);
			break;
		}
	}
}

void TFPolygonModifier::keyPressEvent(QKeyEvent *e){

	switch(e->key())
	{
		case Qt::Key_Alt:
		{
			altPressed_ = true;
			scrollModes_.push_back(ScrollHistogram);
			break;
		}
		case Qt::Key_B:
		{
			if(altPressed_) scrollModes_.push_back(ScrollBase);
			break;
		}
		case Qt::Key_T:
		{
			if(altPressed_) scrollModes_.push_back(ScrollTop);
			break;
		}
	}
}
	
void TFPolygonModifier::keyReleaseEvent(QKeyEvent *e){

	switch(e->key())
	{
		case Qt::Key_Alt:
		{
			altPressed_ = false;
			TF::removeAllFromVector<ScrollMode>(scrollModes_, ScrollHistogram);
			TF::removeAllFromVector<ScrollMode>(scrollModes_, ScrollBase);
			TF::removeAllFromVector<ScrollMode>(scrollModes_, ScrollTop);
			break;
		}
		case Qt::Key_B:
		{
			if(altPressed_) TF::removeAllFromVector<ScrollMode>(scrollModes_, ScrollBase);
			break;
		}
		case Qt::Key_T:
		{
			if(altPressed_) TF::removeAllFromVector<ScrollMode>(scrollModes_, ScrollTop);
			break;
		}
	}
}

void TFPolygonModifier::addPolygon_(const TF::PaintingPoint point){

	addLine_(point.x - baseRadius_, 0,	point.x - topRadius_, point.y);
	addLine_(point.x - topRadius_, point.y, point.x + topRadius_, point.y);
	addLine_(point.x + topRadius_, point.y, point.x + baseRadius_, 0);
}

} // namespace GUI
} // namespace M4D
