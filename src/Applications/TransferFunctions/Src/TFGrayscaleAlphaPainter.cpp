#include "TFGrayscaleAlphaPainter.h"

namespace M4D {
namespace GUI {

TFGrayscaleAlphaPainter::TFGrayscaleAlphaPainter(bool drawAlpha):
	margin_(5),
	spacing_(5),
	colorBarSize_(10),
	background_(Qt::black),
	grey_(Qt::lightGray),
	alpha_(Qt::yellow),
	drawAlpha_(drawAlpha){
}

TFGrayscaleAlphaPainter::~TFGrayscaleAlphaPainter(){}

void TFGrayscaleAlphaPainter::setArea(TFArea area){
	
	area_ = area;

	backgroundArea_= TFArea(
		area_.x,
		area_.y,
		area_.width,
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
}

TFArea TFGrayscaleAlphaPainter::getInputArea(){

	return inputArea_;
}

void TFGrayscaleAlphaPainter::drawBackground(QPainter* drawer){

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

void TFGrayscaleAlphaPainter::drawData(QPainter* drawer, TFColorMapPtr workCopy){

	tfAssert(workCopy->size() == inputArea_.width);

	QColor color;	
	
	TFPaintingPoint origin(inputArea_.x,
		inputArea_.y + inputArea_.height);

	for(TFSize i = 0; i < inputArea_.width - 1; ++i)
	{
		if(drawAlpha_)
		{
			//alpha
			drawer->setPen(alpha_);
			drawer->drawLine(origin.x + i, origin.y - (*workCopy)[i].alpha*inputArea_.height,
				origin.x + i + 1, origin.y - (*workCopy)[i+1].alpha*inputArea_.height);
		}
		//value
		drawer->setPen(grey_);	
		drawer->drawLine(origin.x + i, origin.y - (*workCopy)[i].component1*inputArea_.height,
			origin.x + i + 1, origin.y - (*workCopy)[i+1].component1*inputArea_.height);

		//TODO draw histogram if enabled

		//bottom bar
		color.setRgbF((*workCopy)[i].component1, (*workCopy)[i].component2, (*workCopy)[i].component3, 1);
		drawer->setPen(color);
		drawer->drawLine(bottomBarArea_.x + i, bottomBarArea_.y,
			bottomBarArea_.x + i, bottomBarArea_.y + bottomBarArea_.height - 1);
	}

	//draw last point
	color.setRgbF(
		(*workCopy)[inputArea_.width - 1].component1,
		(*workCopy)[inputArea_.width - 1].component2,
		(*workCopy)[inputArea_.width - 1].component3,
		1);
	drawer->setPen(color);
	drawer->drawLine(
		bottomBarArea_.x + inputArea_.width - 1,
		bottomBarArea_.y,
		bottomBarArea_.x + inputArea_.width - 1,
		bottomBarArea_.y + bottomBarArea_.height - 1);
}

} // namespace GUI
} // namespace M4D
