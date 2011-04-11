#include "TFHSVaPainter.h"

namespace M4D {
namespace GUI {

TFHSVaPainter::TFHSVaPainter(bool drawAlpha):
	TFSimplePainter(Qt::magenta, Qt::cyan, QColor(150,75,0), Qt::yellow){

	drawAlpha_ = drawAlpha;
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
	
	bottomBarArea_= QRect(
		inputArea_.x() - 1,
		area_.height() - colorBarSize_ - 1,
		inputArea_.width() + 2,
		colorBarSize_ + 2);
	
	sideBarArea_= QRect(
		0,
		inputArea_.y(),
		colorBarSize_,
		inputArea_.height());

	sizeChanged_ = true;
}

void TFHSVaPainter::updateSideBar_(WorkCopy::Ptr workCopy){

	viewSideBarBuffer_ = QPixmap(area_.width(), area_.height());
	viewSideBarBuffer_.fill(noColor_);

	QPainter drawer(&viewSideBarBuffer_);

	float zoomedY = 1.0f/workCopy->getZoomY();
	float stepValue = zoomedY/sideBarArea_.height();
	float offset = workCopy->getZoomCenter().y - zoomedY/2.0f;

	QColor color;
	int x1 = sideBarArea_.x();
	int x2 = sideBarArea_.x() + sideBarArea_.width();
	int y1, y2;
	int yBegin = sideBarArea_.y() + sideBarArea_.height();

	for(int i = 0; i < sideBarArea_.height() - 1; ++i)
	{
		y1 = yBegin - i;
		y2 = yBegin - i;

		color.setHsvF(i*stepValue + offset, 1, 1);

		drawer.setPen(color);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

QPixmap TFHSVaPainter::getView(WorkCopy::Ptr workCopy){

	updateSideBar_(workCopy);
	TFSimplePainter::getView(workCopy);

	QPainter drawer(&viewBuffer_);
	drawer.drawPixmap(0, 0, viewSideBarBuffer_);

	return viewBuffer_;
}

} // namespace GUI
} // namespace M4D
