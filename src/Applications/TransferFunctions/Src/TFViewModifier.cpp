#include "TFViewModifier.h"

namespace M4D {
namespace GUI {

TFViewModifier::TFViewModifier(WorkCopy::Ptr workCopy):
	viewTools_(new Ui::TFViewModifier),
	viewWidget_(new QWidget),
	zoomMovement_(false),
	altPressed_(false),
	zoomDirection_(WorkCopy::ZoomX){

	workCopy_ = workCopy;

	viewTools_->setupUi(viewWidget_);

	viewTools_->maxZoomSpin->setValue((int)workCopy_->getMaxZoom());
	viewTools_->zoomXValue->setText(QString::number(workCopy_->getZoomX()));
	viewTools_->zoomYValue->setText(QString::number(workCopy_->getZoomY()));

	TF::Point<float,float> center = workCopy_->getZoomCenter();
	viewTools_->centerXValue->setText(QString::number(center.x));
	viewTools_->centerYValue->setText(QString::number(center.y));

	viewTools_->xAxisCheck->setChecked(true);

	bool histogramCheckConnected = QObject::connect( viewTools_->histogramCheck, SIGNAL(toggled(bool)),
		this, SLOT(histogram_check(bool)));
	tfAssert(histogramCheckConnected);

	bool maxZoomSpinConnected = QObject::connect( viewTools_->maxZoomSpin, SIGNAL(valueChanged(int)),
		this, SLOT(maxZoomSpin_changed(int)));
	tfAssert(maxZoomSpinConnected);

	bool xAxisCheckConnected = QObject::connect( viewTools_->xAxisCheck, SIGNAL(toggled(bool)),
		this, SLOT(xAxis_check(bool)));
	tfAssert(xAxisCheckConnected);
	bool yAxisCheckConnected = QObject::connect( viewTools_->yAxisCheck, SIGNAL(toggled(bool)),
		this, SLOT(yAxis_check(bool)));
	tfAssert(yAxisCheckConnected);
}

TFViewModifier::~TFViewModifier(){}

void TFViewModifier::createTools_(){

	toolsWidget_ = new QWidget();
	toolsWidget_->setLayout(centerWidget_(viewWidget_));
}

QGridLayout* TFViewModifier::centerWidget_(QWidget* widget){

    QGridLayout* centerLayout = new QGridLayout();
    centerLayout->setSpacing(0);
    centerLayout->setObjectName(QString::fromUtf8("centerLayout"));

    QSpacerItem* pushLeftSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);
    centerLayout->addItem(pushLeftSpacer, 1, 2, 1, 1);

    QSpacerItem* pushRightSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);
    centerLayout->addItem(pushRightSpacer, 1, 0, 1, 1);

    QSpacerItem* pushUpSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
    centerLayout->addItem(pushUpSpacer, 2, 1, 1, 1);

    QSpacerItem* pushDownSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
    centerLayout->addItem(pushDownSpacer, 0, 1, 1, 1);

    centerLayout->addWidget(widget, 1, 1, 1, 1);

	return centerLayout;
}
	
bool TFViewModifier::load(TFXmlReader::Ptr reader){

	updateZoomTools_();
	viewTools_->maxZoomSpin->setValue((int)workCopy_->getMaxZoom());
	return true;
}

void TFViewModifier::histogram_check(bool enabled){

	workCopy_->setHistogramEnabled(enabled);
	emit RefreshView();
}

void TFViewModifier::maxZoomSpin_changed(int value){

	workCopy_->setMaxZoom(value);
}

void TFViewModifier::xAxis_check(bool enabled){

	if(enabled)
	{
		if(viewTools_->yAxisCheck->isChecked()) zoomDirection_ = WorkCopy::ZoomBoth;
		else zoomDirection_ = WorkCopy::ZoomX;
	}
	else
	{
		if(viewTools_->yAxisCheck->isChecked()) zoomDirection_ = WorkCopy::ZoomY;
		else zoomDirection_ = WorkCopy::ZoomNone;
	}
}

void TFViewModifier::yAxis_check(bool enabled){

	if(enabled)
	{
		if(viewTools_->xAxisCheck->isChecked()) zoomDirection_ = WorkCopy::ZoomBoth;
		else zoomDirection_ = WorkCopy::ZoomY;
	}
	else
	{
		if(viewTools_->xAxisCheck->isChecked()) zoomDirection_ = WorkCopy::ZoomX;
		else zoomDirection_ = WorkCopy::ZoomNone;
	}
}

void TFViewModifier::updateZoomTools_(){

	viewTools_->zoomXValue->setText(QString::number(workCopy_->getZoomX()));
	viewTools_->zoomYValue->setText(QString::number(workCopy_->getZoomY()));

	TF::Point<float,float> center = workCopy_->getZoomCenter();

	viewTools_->centerXValue->setText(QString::number(center.x));
	viewTools_->centerYValue->setText(QString::number(center.y));
}

void TFViewModifier::mousePressEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y());
	if(relativePoint == ignorePoint_) return;

	if(e->button() == Qt::MidButton)
	{
		zoomMovement_ = true;
		zoomMoveHelper_ = relativePoint;
	}
}

void TFViewModifier::mouseReleaseEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	zoomMovement_ = false;

	emit RefreshView();
}

void TFViewModifier::mouseMoveEvent(QMouseEvent *e){
	
	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(zoomMovement_)
	{
		workCopy_->move(zoomMoveHelper_.x - relativePoint.x, zoomMoveHelper_.y - relativePoint.y);
		zoomMoveHelper_ = relativePoint;
		updateZoomTools_();
	}

	emit RefreshView();
}

void TFViewModifier::wheelEvent(QWheelEvent *e){

	int steps = e->delta() / 120;
	if(steps == 0) return;

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y());
	if(relativePoint == ignorePoint_) return;

	if(altPressed_)
	{
		if(steps > 0) workCopy_->increaseHistogramLogBase(steps);
		if(steps < 0) workCopy_->decreaseHistogramLogBase(-steps);
		emit RefreshView();
		return;
	}

	if(steps > 0) workCopy_->zoomIn(steps, relativePoint.x, relativePoint.y, zoomDirection_);
	if(steps < 0) workCopy_->zoomOut(-steps, relativePoint.x, relativePoint.y, zoomDirection_);
	
	updateZoomTools_();
	emit RefreshView();
}

void TFViewModifier::keyPressEvent(QKeyEvent *e){

	if(e->key() == Qt::Key_Alt) altPressed_ = true;
}
	
void TFViewModifier::keyReleaseEvent(QKeyEvent *e){

	if(e->key() == Qt::Key_Alt) altPressed_ = false;
}

} // namespace GUI
} // namespace M4D
