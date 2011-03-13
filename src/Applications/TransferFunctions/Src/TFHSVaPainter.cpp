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

void TFHSVaPainter::setArea(QRect area){
	
	area_ = area;

	backgroundArea_= QRect(
		colorBarSize_ + spacing_,
		0,
		area_.width() - colorBarSize_ - spacing_,
		area_.height() - colorBarSize_ - spacing_);

	inputArea_= QRect(
		backgroundArea_.x() + margin_,
		backgroundArea_.y() + margin_,
		backgroundArea_.width() - 2*margin_,
		backgroundArea_.height() - 2*margin_);
	
	bottomBarArea_= QRect(	//+1 in each direction as border
		inputArea_.x() - 1,
		area_.height() - colorBarSize_ - 1,
		inputArea_.width() + 2,
		colorBarSize_ + 2);
	
	sideBarArea_= QRect(
		0,
		inputArea_.y(),
		colorBarSize_,
		inputArea_.height());
}

QRect TFHSVaPainter::getInputArea(){

	return QRect(area_.x() + inputArea_.x(), area_.y() + inputArea_.y(),
		inputArea_.width(), inputArea_.height());
}

void TFHSVaPainter::drawBackground_(QPainter* drawer){

	drawSideColorBar_(drawer);

	drawer->fillRect(backgroundArea_, QBrush(background_));

	drawer->fillRect(bottomBarArea_, QBrush(background_));
}

void TFHSVaPainter::drawSideColorBar_(QPainter *drawer){

	QColor color;
	for(int i = 0; i < sideBarArea_.height(); ++i)
	{
		color.setHsvF(i/(float)sideBarArea_.height(), 1, 1);		

		drawer->setPen(color);
		drawer->drawLine(sideBarArea_.x(), sideBarArea_.y() + sideBarArea_.height() - i,
			sideBarArea_.x() + sideBarArea_.width(), sideBarArea_.y() + sideBarArea_.height() - i);
	}
}

void TFHSVaPainter::drawData(QPainter* drawer, TFWorkCopy::Ptr workCopy){

	tfAssert(workCopy->size() == inputArea_.width());

	if(workCopy->changed())
	{
		dBuffer_ = QPixmap(area_.width(), area_.height());

		QPainter dbPainter(&dBuffer_);
		//dbPainter.setBackgroundMode(Qt::TransparentMode);
		
		drawBackground_(&dbPainter);

		dbPainter.setClipRect(inputArea_.x(), inputArea_.y(),
			inputArea_.width() + 1, inputArea_.height() + 1);

		TFPaintingPoint origin(inputArea_.x(), inputArea_.y());
		TFColor tfColor;
		QColor qColor;
		for(int i = 0; i < inputArea_.width() - 1; ++i)
		{
			if(drawAlpha_)
			{
				//alpha
				dbPainter.setPen(alpha_);
				dbPainter.drawLine(origin.x + i, origin.y + (1 - workCopy->getAlpha(i))*inputArea_.height(),
					origin.x + i + 1, origin.y + (1 - workCopy->getAlpha(i+1))*inputArea_.height());
			}
			//value
			dbPainter.setPen(value_);	
			dbPainter.drawLine(origin.x + i, origin.y + (1 - workCopy->getComponent3(i))*inputArea_.height(),
				origin.x + i + 1, origin.y + (1 - workCopy->getComponent3(i+1))*inputArea_.height());
			//saturation
			dbPainter.setPen(saturation_);
			dbPainter.drawLine(origin.x + i, origin.y + (1 - workCopy->getComponent2(i))*inputArea_.height(),
				origin.x + i + 1, origin.y + (1 - workCopy->getComponent2(i+1))*inputArea_.height());
			//hue
			dbPainter.setPen(hue_);
			dbPainter.drawLine(origin.x + i, origin.y + (1 - workCopy->getComponent1(i))*inputArea_.height(),
				origin.x + i + 1, origin.y + (1 - workCopy->getComponent1(i+1))*inputArea_.height());	

			//TODO draw histogram if enabled
		}

		dbPainter.setClipRect(bottomBarArea_.x() + 1, bottomBarArea_.y() + 1,
			bottomBarArea_.width() + 1, bottomBarArea_.height());

		for(int i = 0; i < inputArea_.width(); ++i)
		{
			//bottom bar
			tfColor = workCopy->getColor(i);
			qColor.setRgbF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
			dbPainter.setPen(qColor);

			dbPainter.drawLine(bottomBarArea_.x() + i + 1, bottomBarArea_.y(),
				bottomBarArea_.x() + i + 1, bottomBarArea_.y() + bottomBarArea_.height() - 1);
		}
	}

	drawer->drawPixmap(area_.x(), area_.y(), dBuffer_);
}

} // namespace GUI
} // namespace M4D
