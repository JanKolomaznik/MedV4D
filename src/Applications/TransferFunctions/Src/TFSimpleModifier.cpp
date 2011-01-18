#include "TFSimpleModifier.h"

namespace M4D {
namespace GUI {

TFSimpleModifier::TFSimpleModifier(TFAbstractModifier::Type type, const TFSize& domain):
	type_(type),
	activeView_(Active1),
	inputHelper_(),
	leftMousePressed_(false){

	workCopy_ = TFWorkCopy::Ptr(new TFWorkCopy(domain));
}

TFSimpleModifier::~TFSimpleModifier(){}

void TFSimpleModifier::mousePress(const TFSize& x, const TFSize& y, MouseButton button){

	if(button == MouseButtonMid) return;
	if(button == MouseButtonRight)
	{
		switch(activeView_)
		{
			case Active1:
			{
				activeView_ = active1Next_();
				break;
			}
			case Active2:
			{
				activeView_ = active2Next_();
				break;
			}
			case Active3:
			{
				activeView_ = active3Next_();
				break;
			}
			case ActiveAlpha:
			{
				activeView_ = activeAlphaNext_();
				break;
			}
		}
		return;
	}

	leftMousePressed_ = true;
	inputHelper_.x = x;
	inputHelper_.y = y;
}

void TFSimpleModifier::mouseRelease(const TFSize& x, const TFSize& y){

	if(!leftMousePressed_) return;

	addPoint_(x, y);
	leftMousePressed_ = false;
}

void TFSimpleModifier::mouseMove(const TFSize& x, const TFSize& y){

	if(!leftMousePressed_) return;

	addLine_(inputHelper_.x, inputHelper_.y, x, y);

	inputHelper_.x = x;
	inputHelper_.y = y;
}

void TFSimpleModifier::addPoint_(const TFSize& x, const TFSize& y){

	TFPaintingPoint point = getRelativePoint_(x, y);
	float yValue = point.y/(float)inputArea_.height;
	
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

TFSimpleModifier::ActiveView TFSimpleModifier::active1Next_(){

	switch(type_)
	{
		case TFModifierGrayscale:
		{
			return Active1;
		}
		case TFModifierGrayscaleAlpha:
		{
			return ActiveAlpha;
		}
	}
	return Active2;
}

TFSimpleModifier::ActiveView TFSimpleModifier::active2Next_(){

	switch(type_)
	{
		case TFModifierGrayscale:
		{
			return Active1;
		}
		case TFModifierGrayscaleAlpha:
		{
			return ActiveAlpha;
		}
	}
	return Active3;
}

TFSimpleModifier::ActiveView TFSimpleModifier::active3Next_(){

	if(type_ == TFModifierGrayscale ||
		type_ == TFModifierRGB ||
		type_ == TFModifierHSV)
	{
		return Active1;
	}
	return ActiveAlpha;
}

TFSimpleModifier::ActiveView TFSimpleModifier::activeAlphaNext_(){

	return Active1;
}

} // namespace GUI
} // namespace M4D
