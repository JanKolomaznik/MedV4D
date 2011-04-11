#include "TFSimpleModifier.h"

namespace M4D {
namespace GUI {

TFSimpleModifier::TFSimpleModifier(WorkCopy::Ptr workCopy, Mode mode, bool alpha):
	TFViewModifier(workCopy),
	mode_(mode),
	alpha_(alpha),
	simpleTools_(new Ui::TFSimpleModifier),
	simpleWidget_(new QWidget),
	activeView_(Active1),
	inputHelper_(),
	leftMousePressed_(false){

	simpleTools_->setupUi(simpleWidget_);

	bool changeViewConnected = QObject::connect(simpleTools_->activeViewBox, SIGNAL(currentIndexChanged(int)),
		this, SLOT(activeView_changed(int)));
	tfAssert(changeViewConnected);

	switch(mode_)
	{
		case Grayscale:
		{
			simpleTools_->activeViewBox->addItem(QObject::tr("gray"));
			break;
		}
		case RGB:
		{
			simpleTools_->activeViewBox->addItem(QObject::tr("red"));
			simpleTools_->activeViewBox->addItem(QObject::tr("green"));
			simpleTools_->activeViewBox->addItem(QObject::tr("blue"));
			break;
		}
		case HSV:
		{
			simpleTools_->activeViewBox->addItem(QObject::tr("hue"));
			simpleTools_->activeViewBox->addItem(QObject::tr("saturation"));
			simpleTools_->activeViewBox->addItem(QObject::tr("value"));
			break;
		}
		default:
		{
			tfAssert(!"Painter not supported");
		}
	}
	if(alpha_) simpleTools_->activeViewBox->addItem(QObject::tr("opacity"));
}

TFSimpleModifier::~TFSimpleModifier(){}

void TFSimpleModifier::createTools_(){

    QFrame* separator = new QFrame();
    separator->setFrameShape(QFrame::HLine);
    separator->setFrameShadow(QFrame::Sunken);

	QVBoxLayout* layout = new QVBoxLayout();
	layout->addItem(centerWidget_(simpleWidget_));
	layout->addWidget(separator);
	layout->addItem(centerWidget_(viewWidget_));

	toolsWidget_ = new QWidget();
	toolsWidget_->setLayout(layout);
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

void TFSimpleModifier::mousePressEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y());
	if(relativePoint == ignorePoint_) return;

	if(e->button() == Qt::RightButton)
	{
		int nextIndex = (simpleTools_->activeViewBox->currentIndex()+1) % simpleTools_->activeViewBox->count();
		simpleTools_->activeViewBox->setCurrentIndex(nextIndex);
	}
	if(e->button() == Qt::LeftButton)
	{
		leftMousePressed_ = true;
		inputHelper_ = relativePoint;
	}

	TFViewModifier::mousePressEvent(e);
}

void TFSimpleModifier::mouseReleaseEvent(QMouseEvent *e){

	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_) addPoint_(relativePoint.x, relativePoint.y);
	leftMousePressed_ = false;

	TFViewModifier::mouseReleaseEvent(e);
}

void TFSimpleModifier::mouseMoveEvent(QMouseEvent *e){
	
	TF::PaintingPoint relativePoint = getRelativePoint_(e->x(), e->y(), leftMousePressed_ || zoomMovement_);
	if(relativePoint == ignorePoint_) return;

	if(leftMousePressed_)
	{
		addLine_(inputHelper_, relativePoint);
		inputHelper_ = relativePoint;
	}

	TFViewModifier::mouseMoveEvent(e);
}

void TFSimpleModifier::addPoint_(const int x, const int y){

	float yValue = y/(float)inputArea_.height();
	
	switch(activeView_)
	{
		case Active1:
		{
			workCopy_->setComponent1(x, TF_DIMENSION_1, yValue);
			if(mode_ == Grayscale)
			{
				workCopy_->setComponent2(x, TF_DIMENSION_1, yValue);
				workCopy_->setComponent3(x, TF_DIMENSION_1, yValue);
			}
			break;
		}
		case Active2:
		{
			workCopy_->setComponent2(x, TF_DIMENSION_1, yValue);
			break;
		}
		case Active3:
		{
			workCopy_->setComponent3(x, TF_DIMENSION_1, yValue);
			break;
		}
		case ActiveAlpha:
		{
			workCopy_->setAlpha(x, TF_DIMENSION_1, yValue);
			break;
		}
	}
	changed_ = true;
	emit RefreshView();
}

} // namespace GUI
} // namespace M4D
