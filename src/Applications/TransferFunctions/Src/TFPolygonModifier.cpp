#include "TFPolygonModifier.h"

namespace M4D {
namespace GUI {

TFPolygonModifier::TFPolygonModifier(TFAbstractModifier::Type type, const TFSize& domain):
	type_(type),
	tools_(new Ui::TFPolygonModifier),
	activeView_(Active1),
	inputHelper_(),
	leftMousePressed_(false),
	baseRadius_(50),
	topRadius_(20){

	workCopy_ = TFWorkCopy::Ptr(new TFWorkCopy(domain));

	toolsWidget_ = new QWidget();
	tools_->setupUi(toolsWidget_);

	bool changeViewConnected = QObject::connect(tools_->activeViewBox, SIGNAL(currentIndexChanged(int)),
		this, SLOT(activeViewChanged(int)));
	tfAssert(changeViewConnected);
	bool histogramCheckConnected = QObject::connect(tools_->histogramCheck, SIGNAL(toggled(bool)),
		this, SLOT(histogramCheck(bool)));
	tfAssert(histogramCheckConnected);
	bool topSpinConnected = QObject::connect(tools_->topSpin, SIGNAL(valueChanged(int)),
		this, SLOT(topSpinChanged(int)));
	tfAssert(topSpinConnected);
	bool bottomSpinConnected = QObject::connect(tools_->bottomSpin, SIGNAL(valueChanged(int)),
		this, SLOT(bottomSpinChanged(int)));
	tfAssert(bottomSpinConnected);

	switch(type_)
	{
		case TFModifierGrayscale:
		{
			tools_->activeViewBox->addItem(QObject::tr("gray"));
			break;
		}
		case TFModifierGrayscaleAlpha:
		{
			tools_->activeViewBox->addItem(QObject::tr("gray"));
			tools_->activeViewBox->addItem(QObject::tr("opacity"));
			break;
		}
		case TFModifierRGB:
		{
			tools_->activeViewBox->addItem(QObject::tr("red"));
			tools_->activeViewBox->addItem(QObject::tr("green"));
			tools_->activeViewBox->addItem(QObject::tr("blue"));
			break;
		}
		case TFModifierRGBa:
		{
			tools_->activeViewBox->addItem(QObject::tr("red"));
			tools_->activeViewBox->addItem(QObject::tr("green"));
			tools_->activeViewBox->addItem(QObject::tr("blue"));
			tools_->activeViewBox->addItem(QObject::tr("opacity"));
			break;
		}
		case TFModifierHSV:
		{
			tools_->activeViewBox->addItem(QObject::tr("hue"));
			tools_->activeViewBox->addItem(QObject::tr("saturation"));
			tools_->activeViewBox->addItem(QObject::tr("value"));
			break;
		}
		case TFModifierHSVa:
		{
			tools_->activeViewBox->addItem(QObject::tr("hue"));
			tools_->activeViewBox->addItem(QObject::tr("saturation"));
			tools_->activeViewBox->addItem(QObject::tr("value"));
			tools_->activeViewBox->addItem(QObject::tr("opacity"));
			break;
		}
	}
}

TFPolygonModifier::~TFPolygonModifier(){}

void TFPolygonModifier::histogramCheck(bool enabled){}

void TFPolygonModifier::activeViewChanged(int index){

	switch(index)
	{
		case 0:
		{
			activeView_ = Active1;
			break;
		}
		case 1:
		{
			if(type_ == TFModifierGrayscaleAlpha) activeView_ = ActiveAlpha;
			else activeView_ = Active2;
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
			tfAbort(!"Bad view selected.");
			break;
		}
	}
}

void TFPolygonModifier::mousePress(const TFSize& x, const TFSize& y, MouseButton button){

	if(button == MouseButtonRight)
	{
		int nextIndex = (tools_->activeViewBox->currentIndex()+1) % tools_->activeViewBox->count();
		tools_->activeViewBox->setCurrentIndex(nextIndex);
	}
	if(button == MouseButtonLeft)
	{
		leftMousePressed_ = true;
		inputHelper_.x = x;
		inputHelper_.y = y;
	}
}

void TFPolygonModifier::mouseRelease(const TFSize& x, const TFSize& y){

	if(!leftMousePressed_) return;

	addPolygon_(x, y);
	leftMousePressed_ = false;
}

void TFPolygonModifier::mouseMove(const TFSize& x, const TFSize& y){

	if(!leftMousePressed_) return;

	for(;inputHelper_.x < x; ++inputHelper_.x)
	{
		addPoint_(inputHelper_.x - baseRadius_, inputArea_.y() + inputArea_.height());
	}

	addPolygon_(x, y);

	for(;inputHelper_.x > x; --inputHelper_.x)
	{
		addPoint_(inputHelper_.x + baseRadius_, inputArea_.y() + inputArea_.height());
	}
}

void TFPolygonModifier::addPolygon_(const int &x, const int &y){

	addLine_(x - baseRadius_, inputArea_.y() + inputArea_.height(),	x - topRadius_, y);
	addLine_(x - topRadius_, y, x + topRadius_, y);
	addLine_(x + topRadius_, y, x + baseRadius_, inputArea_.y() + inputArea_.height());
}

void TFPolygonModifier::addPoint_(const int& x, const int& y){

	if(x < 0 || x > (int)(inputArea_.x() + inputArea_.width())) return;

	TFPaintingPoint point = getRelativePoint_(x, y);

	float yValue = point.y/(float)inputArea_.height();
	
	switch(activeView_)
	{
		case Active1:
		{
			workCopy_->setComponent1(point.x, yValue);
			if(type_ == TFModifierGrayscale ||
				type_ == TFModifierGrayscaleAlpha)
			{
				workCopy_->setComponent2(point.x, yValue);
				workCopy_->setComponent3(point.x, yValue);
			}
			break;
		}
		case Active2:
		{
			workCopy_->setComponent2(point.x, yValue);
			break;
		}
		case Active3:
		{
			workCopy_->setComponent3(point.x, yValue);
			break;
		}
		case ActiveAlpha:
		{
			workCopy_->setAlpha(point.x, yValue);
			break;
		}
	}
	++lastChange_;	
}

void TFPolygonModifier::topSpinChanged(int value){

	topRadius_ = value;
	if(baseRadius_ < topRadius_) tools_->bottomSpin->setValue(topRadius_);
}

void TFPolygonModifier::bottomSpinChanged(int value){

	baseRadius_ = value;
	if(baseRadius_ < topRadius_) tools_->topSpin->setValue(baseRadius_);
}

} // namespace GUI
} // namespace M4D
