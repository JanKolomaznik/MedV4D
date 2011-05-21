#include "TFViewModifier.h"

namespace M4D {
namespace GUI {

TFViewModifier::TFViewModifier(TFFunctionInterface::Ptr function, TFAbstractPainter::Ptr painter):
	TFAbstractModifier(function, painter),
	viewTools_(new Ui::TFViewModifier),
	viewWidget_(new QWidget),
	zoomMovement_(false),
	altPressed_(false){

	viewTools_->setupUi(viewWidget_);

	viewTools_->maxZoomSpin->setValue((int)workCopy_->getMaxZoom());

	bool maxZoomSpinConnected = QObject::connect( viewTools_->maxZoomSpin, SIGNAL(valueChanged(int)),
		this, SLOT(maxZoomSpin_changed(int)));
	tfAssert(maxZoomSpinConnected);
	bool histogramCheckConnected = QObject::connect( viewTools_->histogramCheck, SIGNAL(toggled(bool)),
		this, SLOT(histogram_check(bool)));
		tfAssert(histogramCheckConnected);
	
	TF::Size dimension = workCopy_->getDimension();
	QWidget* dimensionWidget = NULL;
	Ui::TFDimensionZoom* dimensionUi = NULL;
	for(TF::Size i = 1; i <= dimension; ++i)
	{
		dimensionWidget = new QWidget();
		dimensionUi = new Ui::TFDimensionZoom();
		dimensionUi->setupUi(dimensionWidget);

		dimensionUi->dimensionValue->setText(QString::number(i));
		dimensionUi->zoomValue->setText(QString::number(workCopy_->getZoom(i)));
		dimensionUi->centerValue->setText(QString::number(workCopy_->getZoomCenter(i)));

		dimensionsUi_.push_back(dimensionUi);
		viewTools_->valuesLayout->addWidget(dimensionWidget);
	}
	dimensionsUi_[0]->axisCheck->setChecked(true);
	viewWidget_->setMinimumHeight(viewWidget_->height() + dimension*dimensionWidget->minimumHeight());
	viewTools_->zoomWidget->resize(viewWidget_->size());

	setMouseTracking(true);
}

TFViewModifier::~TFViewModifier(){

	for(std::vector<Ui::TFDimensionZoom*>::iterator it = dimensionsUi_.begin(); it != dimensionsUi_.end(); ++it)
	{
		delete *it;
	}
	delete viewTools_;
}

void TFViewModifier::createTools_(){

	toolsWidget_ = new QWidget();
	toolsWidget_->setLayout(centerWidget_(viewWidget_));
}

QGridLayout* TFViewModifier::centerWidget_(QWidget* widget, bool top, bool bottom, bool left, bool right){

    QGridLayout* centerLayout = new QGridLayout();
    centerLayout->setSpacing(0);
    centerLayout->setObjectName(QString::fromUtf8("centerLayout"));

	if(right)
	{
		QSpacerItem* pushLeftSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);
		centerLayout->addItem(pushLeftSpacer, 1, 2, 1, 1);
	}
	if(left)
	{
		QSpacerItem* pushRightSpacer = new QSpacerItem(40, 20, QSizePolicy::Expanding, QSizePolicy::Minimum);
		centerLayout->addItem(pushRightSpacer, 1, 0, 1, 1);
	}
	if(bottom)
	{
		QSpacerItem* pushUpSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
		centerLayout->addItem(pushUpSpacer, 2, 1, 1, 1);
	}
	if(top)
	{
		QSpacerItem* pushDownSpacer = new QSpacerItem(20, 40, QSizePolicy::Minimum, QSizePolicy::Expanding);
		centerLayout->addItem(pushDownSpacer, 0, 1, 1, 1);
	}
    centerLayout->addWidget(widget, 1, 1, 1, 1);

	return centerLayout;
}

void TFViewModifier::setDataStructure(const std::vector<TF::Size>& dataStructure){

	workCopy_->setDataStructure(dataStructure);
}
	
bool TFViewModifier::loadSettings_(TF::XmlReaderInterface* reader){

	updateZoomTools_();
	viewTools_->maxZoomSpin->setValue((int)workCopy_->getMaxZoom());
	return true;
}

void TFViewModifier::histogram_check(bool enabled){

	workCopy_->setHistogramEnabled(enabled);
	update();
}

void TFViewModifier::maxZoomSpin_changed(int value){

	workCopy_->setMaxZoom(value);
}

void TFViewModifier::updateZoomTools_(){

	for(TF::Size i = 1; i <= workCopy_->getDimension(); ++i)
	{
		dimensionsUi_[i-1]->dimensionValue->setText(QString::number(i));
		dimensionsUi_[i-1]->zoomValue->setText(QString::number(workCopy_->getZoom(i)));
		dimensionsUi_[i-1]->centerValue->setText(QString::number(workCopy_->getZoomCenter(i)));
	}
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
}

void TFViewModifier::mouseMoveEvent(QMouseEvent *e){
	
	setFocus();
	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(zoomMovement_)
	{
		std::vector<int> increments = computeZoomMoveIncrements_(
			zoomMoveHelper_.x - relativePoint.x,
			relativePoint.y - zoomMoveHelper_.y
		);

		workCopy_->move(increments);

		zoomMoveHelper_ = relativePoint;
		updateZoomTools_();

		update();
	}
}

void TFViewModifier::wheelEvent(QWheelEvent *e){

	int steps = e->delta() / 120;
	if(steps == 0) return;

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y());
	if(relativePoint == ignorePoint_) return;

	if(altPressed_)
	{
		if(steps < 0) workCopy_->increaseHistogramLogBase(-steps);
		if(steps > 0) workCopy_->decreaseHistogramLogBase(steps);
		update();
		return;
	}

	for(TF::Size i = 1; i <= workCopy_->getDimension(); ++i)
	{
		if(dimensionsUi_[i-1]->axisCheck->isChecked()) workCopy_->zoom(i, relativePoint.x, steps);
	}
	
	updateZoomTools_();
	update();
}

void TFViewModifier::keyPressEvent(QKeyEvent *e){

	if(e->key() == Qt::Key_Alt) altPressed_ = true;
}
	
void TFViewModifier::keyReleaseEvent(QKeyEvent *e){

	if(e->key() == Qt::Key_Alt) altPressed_ = false;
}

TF::PaintingPoint TFViewModifier::getRelativePoint_(const int x, const int y, bool acceptOutOfBounds){	

	int xMax = inputArea_.width() - 1;
	int yMax = inputArea_.height();
	
	TF::PaintingPoint corrected = TF::PaintingPoint(x - inputArea_.x(),
		yMax - (y - inputArea_.y()));

	bool outOfBounds = false;
	if( corrected.x < 0 ||
		corrected.x > xMax ||
		corrected.y < 0 ||
		corrected.y > yMax)
	{
		outOfBounds = true;
	}	
	if(outOfBounds && !acceptOutOfBounds) return ignorePoint_;
	return corrected;
}

} // namespace GUI
} // namespace M4D
