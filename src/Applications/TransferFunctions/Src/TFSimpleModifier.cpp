#include "TFSimpleModifier.h"

namespace M4D {
namespace GUI {

TFSimpleModifier::TFSimpleModifier(TFWorkCopy<TF_SIMPLEMODIFIER_DIMENSION>::Ptr workCopy, Mode mode, bool alpha):
	mode_(mode),
	alpha_(alpha),
	tools_(new Ui::TFSimpleModifier),
	activeView_(Active1),
	inputHelper_(),
	leftMousePressed_(false),
	zoomMovement_(false),
	histScroll_(false){

	workCopy_ = workCopy;

	toolsWidget_ = new QWidget();
	tools_->setupUi(toolsWidget_);

	tools_->maxZoomSpin->setValue((int)workCopy_->getMaxZoom());
	tools_->ratioValue->setText(QString::number(workCopy_->getZoom()));

	TF::Point<float,float> center = workCopy_->getZoomCenter();
	tools_->zoomXValue->setText(QString::number(center.x));
	tools_->zoomYValue->setText(QString::number(center.y));

	bool changeViewConnected = QObject::connect(tools_->activeViewBox, SIGNAL(currentIndexChanged(int)),
		this, SLOT(activeView_changed(int)));
	tfAssert(changeViewConnected);
	bool histogramCheckConnected = QObject::connect( tools_->histogramCheck, SIGNAL(toggled(bool)),
		this, SLOT(histogram_check(bool)));
	tfAssert(histogramCheckConnected);

	bool maxZoomSpinConnected = QObject::connect( tools_->maxZoomSpin, SIGNAL(valueChanged(int)),
		this, SLOT(maxZoomSpin_changed(int)));
	tfAssert(maxZoomSpinConnected);

	switch(mode_)
	{
		case Grayscale:
		{
			tools_->activeViewBox->addItem(QObject::tr("gray"));
			break;
		}
		case RGB:
		{
			tools_->activeViewBox->addItem(QObject::tr("red"));
			tools_->activeViewBox->addItem(QObject::tr("green"));
			tools_->activeViewBox->addItem(QObject::tr("blue"));
			break;
		}
		case HSV:
		{
			tools_->activeViewBox->addItem(QObject::tr("hue"));
			tools_->activeViewBox->addItem(QObject::tr("saturation"));
			tools_->activeViewBox->addItem(QObject::tr("value"));
			break;
		}
		default:
		{
			tfAssert(!"Painter not supported");
		}
	}
	if(alpha_) tools_->activeViewBox->addItem(QObject::tr("opacity"));
}

TFSimpleModifier::~TFSimpleModifier(){}
	
bool TFSimpleModifier::load(TFXmlReader::Ptr reader){

	updateZoomTools_();
	tools_->maxZoomSpin->setValue((int)workCopy_->getMaxZoom());
	return true;
}

void TFSimpleModifier::histogram_check(bool enabled){

	workCopy_->setHistogramEnabled(enabled);
	emit RefreshView();
}

void TFSimpleModifier::activeView_changed(int index){

	switch(index)
	{
		case 0:
		{
			activeView_ = Active1;
			break;
		}
		case 1:
		{
			if(mode_ == Grayscale)
			{
				activeView_ = ActiveAlpha;
			}
			else
			{
				activeView_ = Active2;
			}
			break;
		}
		case 2:
		{
			activeView_ = Active3;
			break;
		}
		case 3:
		{
			activeView_ = ActiveAlpha;
			break;
		}
		default:
		{
			tfAssert(!"Bad view selected.");
			break;
		}
	}
}

void TFSimpleModifier::maxZoomSpin_changed(int value){

	workCopy_->setMaxZoom(value);
}

void TFSimpleModifier::updateZoomTools_(){

	tools_->ratioValue->setText(QString::number(workCopy_->getZoom()));

	TF::Point<float,float> center = workCopy_->getZoomCenter();

	tools_->zoomXValue->setText(QString::number(center.x));
	tools_->zoomYValue->setText(QString::number(center.y));
}

void TFSimpleModifier::mousePress(const int x, const int y, Qt::MouseButton button){

	TF::PaintingPoint relativePoint = getRelativePoint_(x, y);
	if(relativePoint == ignorePoint_) return;

	if(button == Qt::RightButton)
	{
		int nextIndex = (tools_->activeViewBox->currentIndex()+1) % tools_->activeViewBox->count();
		tools_->activeViewBox->setCurrentIndex(nextIndex);
	}
	if(button == Qt::LeftButton)
	{
		leftMousePressed_ = true;
		inputHelper_ = relativePoint;
	}
	if(button == Qt::MidButton)
	{
		zoomMovement_ = true;
		zoomMoveHelper_ = relativePoint;
	}

	emit RefreshView();
}

void TFSimpleModifier::mouseRelease(const int x, const int y){

	TF::PaintingPoint relativePoint = getRelativePoint_(x, y, leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_) addPoint_(relativePoint.x, relativePoint.y);

	leftMousePressed_ = false;
	zoomMovement_ = false;

	emit RefreshView();
}

void TFSimpleModifier::mouseMove(const int x, const int y){
	
	TF::PaintingPoint relativePoint = getRelativePoint_(x, y, leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_)
	{
		addLine_(inputHelper_, relativePoint);
		inputHelper_ = relativePoint;
	}

	if(zoomMovement_)
	{
		workCopy_->move(zoomMoveHelper_.x - relativePoint.x, zoomMoveHelper_.y - relativePoint.y);
		zoomMoveHelper_ = relativePoint;
		updateZoomTools_();
	}

	emit RefreshView();
}

void TFSimpleModifier::mouseWheel(const int steps, const int x, const int y){

	TF::PaintingPoint relativePoint = getRelativePoint_(x,y);
	if(relativePoint == ignorePoint_) return;

	if(histScroll_)
	{
		if(steps > 0) workCopy_->increaseHistogramLogBase(2.0*steps);
		if(steps < 0) workCopy_->decreaseHistogramLogBase(2.0*(-steps));
		emit RefreshView();
		return;
	}

	if(steps > 0) workCopy_->zoomIn(steps, relativePoint.x, relativePoint.y);
	if(steps < 0) workCopy_->zoomOut(-steps, relativePoint.x, relativePoint.y);
	
	updateZoomTools_();
	emit RefreshView();
}

void TFSimpleModifier::keyPress(int qtKey){

	if(qtKey == Qt::Key_Control) histScroll_ = true;
}
	
void TFSimpleModifier::keyRelease(int qtKey){

	if(qtKey == Qt::Key_Control) histScroll_ = false;
}

void TFSimpleModifier::addPoint_(const int x, const int y){

	float yValue = y/(float)inputArea_.height();
	
	switch(activeView_)
	{
		case Active1:
		{
			workCopy_->setComponent1(x, TF_SIMPLEMODIFIER_DIMENSION, yValue);
			if(mode_ == Grayscale)
			{
				workCopy_->setComponent2(x, TF_SIMPLEMODIFIER_DIMENSION, yValue);
				workCopy_->setComponent3(x, TF_SIMPLEMODIFIER_DIMENSION, yValue);
			}
			break;
		}
		case Active2:
		{
			workCopy_->setComponent2(x, TF_SIMPLEMODIFIER_DIMENSION, yValue);
			break;
		}
		case Active3:
		{
			workCopy_->setComponent3(x, TF_SIMPLEMODIFIER_DIMENSION, yValue);
			break;
		}
		case ActiveAlpha:
		{
			workCopy_->setAlpha(x, TF_SIMPLEMODIFIER_DIMENSION, yValue);
			break;
		}
	}
	changed_ = true;;	
}

} // namespace GUI
} // namespace M4D
