#include "TFSimpleModifier.h"

namespace M4D {
namespace GUI {

TFSimpleModifier::TFSimpleModifier(TFAbstractModifier::Type type):
	type_(type),
	activeView_(Active1),
	inputHelper_(),
	leftMousePressed_(false){}

TFSimpleModifier::~TFSimpleModifier(){}

void TFSimpleModifier::mousePress(TFSize x, TFSize y, MouseButton button){

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

void TFSimpleModifier::mouseRelease(TFSize x, TFSize y){

	if(!leftMousePressed_) return;

	addPoint_(x, y);
	leftMousePressed_ = false;
}

void TFSimpleModifier::mouseMove(TFSize x, TFSize y){

	if(!leftMousePressed_) return;

	addLine_(inputHelper_.x, inputHelper_.y, x, y);

	inputHelper_.x = x;
	inputHelper_.y = y;
}

void TFSimpleModifier::addPoint_(TFSize x, TFSize y){

	TFPaintingPoint point = getRelativePoint_(x, y);
	float yValue = point.y/(float)inputArea_.height;
	
	switch(activeView_)
	{
		case Active1:
		{
			(*workCopy_)[point.x].component1 = yValue;
			if(type_ == TFModifierGrayscale ||
				type_ == TFModifierGrayscaleAlpha)
			{
				(*workCopy_)[point.x].component2 = yValue;
				(*workCopy_)[point.x].component3 = yValue;
			}
			break;
		}
		case Active2:
		{
			(*workCopy_)[point.x].component2 = yValue;
			break;
		}
		case Active3:
		{
			(*workCopy_)[point.x].component3 = yValue;
			break;
		}
		case ActiveAlpha:
		{
			(*workCopy_)[point.x].alpha = yValue;
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
