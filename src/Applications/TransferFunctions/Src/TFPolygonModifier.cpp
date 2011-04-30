#include "TFPolygonModifier.h"

namespace M4D {
namespace GUI {

TFPolygonModifier::TFPolygonModifier(
		TFFunctionInterface::Ptr function,
		TFSimplePainter::Ptr painter):
	TFSimpleModifier(function, painter),
	polygonTools_(new Ui::TFPolygonModifier),
	polygonWidget_(new QWidget),
	polygonSpinStep_(10){

	polygonTools_->setupUi(polygonWidget_);

	topRadius_ = polygonTools_->topSpin->value()/2;
	polygonTools_->topSpin->setSingleStep(polygonSpinStep_);

	bool topSpinConnected = QObject::connect(polygonTools_->topSpin, SIGNAL(valueChanged(int)),
		this, SLOT(topSpin_changed(int)));
	tfAssert(topSpinConnected);

	baseRadius_ = polygonTools_->baseSpin->value()/2;
	polygonTools_->baseSpin->setSingleStep(polygonSpinStep_);

	bool baseSpinConnected = QObject::connect(polygonTools_->baseSpin, SIGNAL(valueChanged(int)),
		this, SLOT(baseSpin_changed(int)));
	tfAssert(baseSpinConnected);

	scrollModes_.push_back(ScrollZoom);
}

TFPolygonModifier::~TFPolygonModifier(){

	delete polygonTools_;
}

void TFPolygonModifier::createTools_(){

    QFrame* vSeparator = new QFrame();
    vSeparator->setFrameShape(QFrame::HLine);
    vSeparator->setFrameShadow(QFrame::Sunken);

    QFrame* hSeparator = new QFrame();
    hSeparator->setFrameShape(QFrame::VLine);
    hSeparator->setFrameShadow(QFrame::Sunken);

	QHBoxLayout* hLayout = new QHBoxLayout();
	hLayout->addItem(centerWidget_(simpleWidget_));
	hLayout->addWidget(hSeparator);
	hLayout->addItem(centerWidget_(polygonWidget_));

	QVBoxLayout* vLayout = new QVBoxLayout();
	vLayout->addItem(hLayout);
	vLayout->addWidget(vSeparator);
	vLayout->addItem(centerWidget_(viewWidget_));

	toolsWidget_ = new QWidget();
	toolsWidget_->setLayout(vLayout);
}

void TFPolygonModifier::topSpin_changed(int value){

	topRadius_ = value/2;
	if(baseRadius_ < topRadius_) polygonTools_->baseSpin->setValue(value);
}

void TFPolygonModifier::baseSpin_changed(int value){

	baseRadius_ = value/2;
	if(baseRadius_ < topRadius_) polygonTools_->topSpin->setValue(value);
}

void TFPolygonModifier::mouseReleaseEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_) addPolygon_(relativePoint);
	leftMousePressed_ = false;
	update();

	TFViewModifier::mouseReleaseEvent(e);
}

void TFPolygonModifier::mouseMoveEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_)
	{
		float zoom = workCopy_->getZoom(1);
		for(;inputHelper_.x < relativePoint.x; ++inputHelper_.x)
		{
			addPoint_(inputHelper_.x - baseRadius_*zoom, 0);
		}

		addPolygon_(relativePoint);

		for(;inputHelper_.x > relativePoint.x; --inputHelper_.x)
		{
			addPoint_(inputHelper_.x + baseRadius_*zoom, 0);
		}
		update();
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
			polygonTools_->baseSpin->setValue(polygonTools_->baseSpin->value() + polygonSpinStep_*steps);	
			break;
		}
		case ScrollTop:
		{
			polygonTools_->topSpin->setValue(polygonTools_->topSpin->value() + polygonSpinStep_*steps);	
			break;
		}
		default:
		{			
			TFSimpleModifier::wheelEvent(e);
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

	float zoom = workCopy_->getZoom(1);
	addLine_(point.x - baseRadius_*zoom, 0,	point.x - topRadius_*zoom, point.y);
	addLine_(point.x - topRadius_*zoom, point.y, point.x + topRadius_*zoom, point.y);
	addLine_(point.x + topRadius_*zoom, point.y, point.x + baseRadius_*zoom, 0);
}

} // namespace GUI
} // namespace M4D
