#include "TFPolygonModifier.h"

namespace M4D {
namespace GUI {

TFPolygonModifier::TFPolygonModifier(TFAbstractModifier::Type type, const TFSize& domain):
	type_(type),
	activeView_(Active1),
	inputHelper_(),
	leftMousePressed_(false),
	baseRadius_(50),
	topRadius_(20){

	workCopy_ = TFWorkCopy::Ptr(new TFWorkCopy(domain));
}

TFPolygonModifier::~TFPolygonModifier(){}

void TFPolygonModifier::mousePress(const TFSize& x, const TFSize& y, MouseButton button){

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

void TFPolygonModifier::mouseRelease(const TFSize& x, const TFSize& y){

	if(!leftMousePressed_) return;

	addPolygon_(x, y);

	leftMousePressed_ = false;
}

void TFPolygonModifier::mouseMove(const TFSize& x, const TFSize& y){

	if(!leftMousePressed_) return;

	for(;inputHelper_.x < x; ++inputHelper_.x)
	{
		addPoint_(inputHelper_.x - baseRadius_, inputArea_.y + inputArea_.height);
	}

	addPolygon_(x, y);

	for(;inputHelper_.x > x; --inputHelper_.x)
	{
		addPoint_(inputHelper_.x + baseRadius_, inputArea_.y + inputArea_.height);
	}
}

void TFPolygonModifier::addPolygon_(const int &x, const int &y){

	addLine_(x - baseRadius_, inputArea_.y + inputArea_.height,	x - topRadius_, y);
	addLine_(x - topRadius_, y, x + topRadius_, y);
	addLine_(x + topRadius_, y, x + baseRadius_, inputArea_.y + inputArea_.height);
}

void TFPolygonModifier::addPoint_(const int& x, const int& y){

	if(x < 0 || x > (int)(inputArea_.x + inputArea_.width)) return;

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

TFPolygonModifier::ActiveView TFPolygonModifier::active1Next_(){

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

TFPolygonModifier::ActiveView TFPolygonModifier::active2Next_(){

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

TFPolygonModifier::ActiveView TFPolygonModifier::active3Next_(){

	if(type_ == TFModifierGrayscale ||
		type_ == TFModifierRGB ||
		type_ == TFModifierHSV)
	{
		return Active1;
	}
	return ActiveAlpha;
}

TFPolygonModifier::ActiveView TFPolygonModifier::activeAlphaNext_(){

	return Active1;
}

} // namespace GUI
} // namespace M4D
