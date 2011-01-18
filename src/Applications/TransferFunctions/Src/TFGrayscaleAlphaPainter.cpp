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

const TFArea& TFGrayscaleAlphaPainter::getInputArea(){

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

void TFGrayscaleAlphaPainter::drawData(QPainter* drawer, TFWorkCopy::Ptr workCopy){

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
		drawer->setPen(grey_);	
		drawer->drawLine(origin.x + i, origin.y - workCopy->getComponent1(i)*inputArea_.height,
			origin.x + i + 1, origin.y - workCopy->getComponent1(i+1)*inputArea_.height);

		//TODO draw histogram if enabled

		//bottom bar
		tfColor = workCopy->getColor(i);
		qColor.setRgbF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
		drawer->setPen(qColor);

		drawer->drawLine(bottomBarArea_.x + i, bottomBarArea_.y,
			bottomBarArea_.x + i, bottomBarArea_.y + bottomBarArea_.height - 1);
	}

	//draw last point
	tfColor = workCopy->getColor(inputArea_.width - 1);
	qColor.setRgbF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
	drawer->setPen(qColor);

	drawer->drawLine(
		bottomBarArea_.x + inputArea_.width - 1,
		bottomBarArea_.y,
		bottomBarArea_.x + inputArea_.width - 1,
		bottomBarArea_.y + bottomBarArea_.height - 1);
}

} // namespace GUI
} // namespace M4D
