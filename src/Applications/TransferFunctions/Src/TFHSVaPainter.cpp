#include "TFHSVaPainter.h"

namespace M4D {
namespace GUI {

TFHSVaPainter::TFHSVaPainter(bool drawAlpha):
	margin_(5),
	spacing_(5),
	colorBarSize_(10),
	background_(Qt::black),
	hue_(Qt::darkMagenta),
	saturation_(Qt::darkCyan),
	value_(Qt::lightGray),
	alpha_(Qt::yellow),
	drawAlpha_(drawAlpha){
}

TFHSVaPainter::~TFHSVaPainter(){}

void TFHSVaPainter::setArea(TFArea area){
	
	area_ = area;

	backgroundArea_= TFArea(
		area_.x + colorBarSize_ + spacing_,
		area_.y,
		area_.width - colorBarSize_ - spacing_,
		area_.height - colorBarSize_ - spacing_);

	inputArea_= TFArea(
		backgroundArea_.x + margin_,
		backgroundArea_.y + margin_,
		backgroundArea_.width - 2*margin_,
		backgroundArea_.height - 2*margin_);
	
	bottomBarArea_= TFArea(
		inputArea_.x,
		area_.y + area_.height - colorBarSize_,
		inputArea_.width,
		colorBarSize_);
	
	sideBarArea_= TFArea(
		area_.x,
		inputArea_.y,
		colorBarSize_,
		inputArea_.height);
}

const TFArea& TFHSVaPainter::getInputArea(){

	return inputArea_;
}

void TFHSVaPainter::drawBackground(QPainter* drawer){

	drawSideColorBar_(drawer);

	QRect paintingRect(
		backgroundArea_.x,
		backgroundArea_.y,
		backgroundArea_.width,
		backgroundArea_.height);

	drawer->fillRect(paintingRect, QBrush(background_));

	QRect bottomBarRect(	//+1 in each direction as border
		bottomBarArea_.x - 1,
		bottomBarArea_.y - 1,
		bottomBarArea_.width + 2,
		bottomBarArea_.height + 2);

	drawer->fillRect(bottomBarRect, QBrush(background_));
}

void TFHSVaPainter::drawSideColorBar_(QPainter *drawer){

	QColor color;
	for(TFSize i = 0; i < sideBarArea_.height; ++i)
	{
		color.setHsvF(i/(float)sideBarArea_.height, 1, 1);		

		drawer->setPen(color);
		drawer->drawLine(sideBarArea_.x, sideBarArea_.y + sideBarArea_.height - i,
			sideBarArea_.x + sideBarArea_.width, sideBarArea_.y + sideBarArea_.height - i);
	}
}

void TFHSVaPainter::drawData(QPainter* drawer, TFWorkCopy::Ptr workCopy){

	tfAssert(workCopy->size() == inputArea_.width);
	
	TFPaintingPoint origin(inputArea_.x,
		inputArea_.y + inputArea_.height);

	TFColor tfColor;
	QColor qColor;

	for(TFSize i = 0; i < inputArea_.width - 1; ++i)
	{
		if(drawAlpha_)
		{
			//alpha
			drawer->setPen(alpha_);
			drawer->drawLine(origin.x + i, origin.y - workCopy->getAlpha(i)*inputArea_.height,
				origin.x + i + 1, origin.y - workCopy->getAlpha(i+1)*inputArea_.height);
		}
		//value
		drawer->setPen(value_);	
		drawer->drawLine(origin.x + i, origin.y - workCopy->getComponent3(i)*inputArea_.height,
			origin.x + i + 1, origin.y - workCopy->getComponent3(i+1)*inputArea_.height);
		//saturation
		drawer->setPen(saturation_);
		drawer->drawLine(origin.x + i, origin.y - workCopy->getComponent2(i)*inputArea_.height,
			origin.x + i + 1, origin.y - workCopy->getComponent2(i+1)*inputArea_.height);
		//hue
		drawer->setPen(hue_);
		drawer->drawLine(origin.x + i, origin.y - workCopy->getComponent1(i)*inputArea_.height,
			origin.x + i + 1, origin.y - workCopy->getComponent1(i+1)*inputArea_.height);		

		//TODO draw histogram if enabled

		//bottom bar
		tfColor = workCopy->getColor(i);
		qColor.setHsvF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
		drawer->setPen(qColor);

		drawer->drawLine(bottomBarArea_.x + i, bottomBarArea_.y,
			bottomBarArea_.x + i, bottomBarArea_.y + bottomBarArea_.height - 1);
	}

	//draw last point
	tfColor = workCopy->getColor(inputArea_.width - 1);
	qColor.setHsvF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
	drawer->setPen(qColor);

	drawer->drawLine(
		bottomBarArea_.x + inputArea_.width - 1,
		bottomBarArea_.y,
		bottomBarArea_.x + inputArea_.width - 1,
		bottomBarArea_.y + bottomBarArea_.height - 1);
}

} // namespace GUI
} // namespace M4D
