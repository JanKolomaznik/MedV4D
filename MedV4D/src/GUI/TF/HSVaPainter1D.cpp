#include "MedV4D/GUI/TF/HSVaPainter1D.h"

namespace M4D {
namespace GUI {

HSVaPainter1D::HSVaPainter1D():
	Painter1D(Qt::magenta, Qt::cyan, Qt::yellow){

	componentNames_.push_back("Hue");
	componentNames_.push_back("Saturation");
	componentNames_.push_back("Value");
	componentNames_.push_back("Opacity");
}


HSVaPainter1D::HSVaPainter1D(const QColor& hue,
							 const QColor& saturation,
							 const QColor& value,
							 const QColor& alpha):
	Painter1D(hue, saturation, value, alpha){

	componentNames_.push_back("Hue");
	componentNames_.push_back("Saturation");
	componentNames_.push_back("Value");
	componentNames_.push_back("Opacity");
}

HSVaPainter1D::~HSVaPainter1D(){}

void HSVaPainter1D::setArea(QRect area){
	
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

void HSVaPainter1D::updateSideBar_(WorkCopy::Ptr workCopy){

	viewSideBarBuffer_ = QPixmap(area_.width(), area_.height());
	viewSideBarBuffer_.fill(noColor_);

	QPainter drawer(&viewSideBarBuffer_);

	QColor color;
	int x1 = sideBarArea_.x();
	int x2 = sideBarArea_.x() + sideBarArea_.width();
	int y1, y2;
	int yBegin = sideBarArea_.y() + sideBarArea_.height();

	for(int i = 0; i < sideBarArea_.height() - 1; ++i)
	{
		y1 = yBegin - i;
		y2 = yBegin - i;

		color.setHsvF(i/(float)sideBarArea_.height(), 1, 1);

		drawer.setPen(color);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

QPixmap HSVaPainter1D::getView(WorkCopy::Ptr workCopy){

	if(sizeChanged_) updateSideBar_(workCopy);
	Painter1D::getView(workCopy);

	QPainter drawer(&viewBuffer_);
	drawer.drawPixmap(0, 0, viewSideBarBuffer_);

	return viewBuffer_;
}

} // namespace GUI
} // namespace M4D
