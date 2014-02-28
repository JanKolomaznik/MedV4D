#include "MedV4D/GUI/TF/Painter1D.h"

namespace M4D {
namespace GUI {

Painter1D::Painter1D(const QColor& component1,
								 const QColor& component2,
								 const QColor& component3,
								 const QColor& alpha):	
	margin_(5),
	spacing_(5),
	colorBarSize_(10),
	background_(Qt::black),
	component1_(component1),
	component2_(component2),
	component3_(component3),
	alpha_(alpha),
	hist_(Qt::darkGray),
	noColor_(0,0,0,0),
	sizeChanged_(true){
}

Painter1D::~Painter1D(){}

void Painter1D::setArea(QRect area){
	
	area_ = area;

	backgroundArea_= QRect(
		0,
		0,
		area_.width(),
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

	sizeChanged_ = true;
}

QRect Painter1D::getInputArea(){

	return inputArea_;
}

std::vector<std::string> Painter1D::getComponentNames(){

	return componentNames_;
}

void Painter1D::updateBackground_(){

	viewBackgroundBuffer_ = QPixmap(area_.width(), area_.height());
	viewBackgroundBuffer_.fill(noColor_);

	QPainter drawer(&viewBackgroundBuffer_);
	drawer.fillRect(backgroundArea_, QBrush(background_));
	drawer.fillRect(bottomBarArea_, QBrush(background_));
}

void Painter1D::updateHistogramView_(WorkCopy::Ptr workCopy){
		
	viewHistogramBuffer_ = QPixmap(area_.width(), area_.height());
	viewHistogramBuffer_.fill(noColor_);

	if(!workCopy->histogramEnabled()) return;

	QPainter drawer(&viewHistogramBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2;
	int y2 = origin.y + inputArea_.height();
	TF::Coordinates coords(1, 0);
	for(int i = 0; i < inputArea_.width(); ++i)
	{
		coords[0] = i;
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getHistogramValue(coords))*inputArea_.height();
		x2 = origin.x + i;

		drawer.setPen(hist_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void Painter1D::updateFunctionView_(WorkCopy::Ptr workCopy){
		
	viewFunctionBuffer_ = QPixmap(area_.width(), area_.height());
	viewFunctionBuffer_.fill(noColor_);

	QPainter drawer(&viewFunctionBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	TF::Coordinates coordsBegin(1, 0);
	TF::Coordinates coordsEnd(1, 0);
	TF::Color colorBegin;
	TF::Color colorEnd;
	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		coordsBegin[0] = i;
		coordsEnd[0] = i + 1;
		colorBegin = workCopy->getColor(coordsBegin);
		colorEnd = workCopy->getColor(coordsEnd);

		x1 = origin.x + coordsBegin[0];
		x2 = origin.x + coordsEnd[0];

		y1 = origin.y + (1 - colorBegin.component1)*inputArea_.height();
		y2 = origin.y + (1 - colorEnd.component1)*inputArea_.height();
		drawer.setPen(component1_);
		drawer.drawLine(x1, y1,	x2, y2);

		y1 = origin.y + (1 - colorBegin.component2)*inputArea_.height();
		y2 = origin.y + (1 - colorEnd.component2)*inputArea_.height();
		drawer.setPen(component2_);
		drawer.drawLine(x1, y1,	x2, y2);	

		y1 = origin.y + (1 - colorBegin.component3)*inputArea_.height();
		y2 = origin.y + (1 - colorEnd.component3)*inputArea_.height();
		drawer.setPen(component3_);
		drawer.drawLine(x1, y1,	x2, y2);	

		y1 = origin.y + (1 - colorBegin.alpha)*inputArea_.height();
		y2 = origin.y + (1 - colorEnd.alpha)*inputArea_.height();
		drawer.setPen(alpha_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}
/*
void Painter1D::updateComponent2View_(WorkCopy::Ptr workCopy){
		
	viewComponent2Buffer_ = QPixmap(area_.width(), area_.height());
	viewComponent2Buffer_.fill(noColor_);

	QPainter drawer(&viewComponent2Buffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent2(TF_DIMENSION_1, i))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent2(TF_DIMENSION_1, i + 1))*inputArea_.height();

		drawer.setPen(component2_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void Painter1D::updateComponent3View_(WorkCopy::Ptr workCopy){
		
	viewComponent3Buffer_ = QPixmap(area_.width(), area_.height());
	viewComponent3Buffer_.fill(noColor_);

	QPainter drawer(&viewComponent3Buffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent3(TF_DIMENSION_1, i))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent3(TF_DIMENSION_1, i + 1))*inputArea_.height();

		drawer.setPen(component3_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void Painter1D::updateAlphaView_(WorkCopy::Ptr workCopy){
		
	viewAlphaBuffer_ = QPixmap(area_.width(), area_.height());
	viewAlphaBuffer_.fill(noColor_);

	QPainter drawer(&viewAlphaBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getAlpha(TF_DIMENSION_1, i))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getAlpha(TF_DIMENSION_1, i + 1))*inputArea_.height();

		drawer.setPen(alpha_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}
*/
void Painter1D::updateBottomColorBarView_(WorkCopy::Ptr workCopy){
		
	viewBottomColorBarBuffer_ = QPixmap(area_.width(), area_.height());
	viewBottomColorBarBuffer_.fill(noColor_);

	QPainter drawer(&viewBottomColorBarBuffer_);
	drawer.setClipRect(bottomBarArea_.x() + 1, bottomBarArea_.y() + 1,
		bottomBarArea_.width() + 1, bottomBarArea_.height());

	TF::Coordinates coords(1);
	TF::Color tfColor;
	QColor qColor;
	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width(); ++i)
	{
		coords[0] = i;
		tfColor = workCopy->getRGBfColor(coords);

		qColor.setRgbF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
		drawer.setPen(qColor);
		
		x1 = bottomBarArea_.x() + i + 1;
		y1 = bottomBarArea_.y();
		x2 = bottomBarArea_.x() + i + 1;
		y2 = bottomBarArea_.y() + bottomBarArea_.height() - 1;

		drawer.drawLine(x1, y1, x2, y2);
	}
}

QPixmap Painter1D::getView(WorkCopy::Ptr workCopy){

	bool change = false;
	if(sizeChanged_)
	{
		updateBackground_();
		updateHistogramView_(workCopy);
		updateFunctionView_(workCopy);
		change = true;
	}
	else
	{
		if(workCopy->histogramChanged())
		{
			updateHistogramView_(workCopy);
			change = true;
		}
		if(workCopy->changed())
		{
			updateFunctionView_(workCopy);
			change = true;
		}
	}
	if(change)
	{
		updateBottomColorBarView_(workCopy);

		viewBuffer_ = QPixmap(area_.width(), area_.height());
		viewBuffer_.fill(noColor_);
		QPainter drawer(&viewBuffer_);

		drawer.drawPixmap(0, 0, viewBackgroundBuffer_);
		drawer.drawPixmap(0, 0, viewHistogramBuffer_);
		drawer.drawPixmap(0, 0, viewFunctionBuffer_);
		drawer.drawPixmap(0, 0, viewBottomColorBarBuffer_);
	}

	sizeChanged_ = false;
	return viewBuffer_;
}

} // namespace GUI
} // namespace M4D
