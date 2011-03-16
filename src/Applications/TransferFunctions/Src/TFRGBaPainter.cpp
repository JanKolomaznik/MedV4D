#include "TFRGBaPainter.h"

namespace M4D {
namespace GUI {

TFRGBaPainter::TFRGBaPainter(bool drawAlpha):
	margin_(5),
	spacing_(5),
	colorBarSize_(10),
	background_(Qt::black),
	red_(Qt::red),
	green_(Qt::green),
	blue_(Qt::blue),
	alpha_(Qt::yellow),
	hist_(255,140,0,255),
	noColor_(0,0,0,0),
	drawAlpha_(drawAlpha),
	sizeChanged_(true){
}

TFRGBaPainter::~TFRGBaPainter(){}

void TFRGBaPainter::setArea(QRect area){
	
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
	
	bottomBarArea_= QRect(	//+1 in each direction as border
		inputArea_.x() - 1,
		area_.height() - colorBarSize_ - 1,
		inputArea_.width() + 2,
		colorBarSize_ + 2);

	sizeChanged_ = true;
}

QRect TFRGBaPainter::getInputArea(){

	return QRect(area_.x() + inputArea_.x(), area_.y() + inputArea_.y(),
		inputArea_.width(), inputArea_.height());
}

void TFRGBaPainter::updateBackground_(){

	viewBackgroundBuffer_ = QPixmap(area_.width(), area_.height());
	viewBackgroundBuffer_.fill(noColor_);

	QPainter drawer(&viewBackgroundBuffer_);
	drawer.fillRect(backgroundArea_, QBrush(background_));
	drawer.fillRect(bottomBarArea_, QBrush(background_));
}

void TFRGBaPainter::updateHistogramView_(TFWorkCopy::Ptr workCopy){
		
	viewHistogramBuffer_ = QPixmap(area_.width(), area_.height());
	viewHistogramBuffer_.fill(noColor_);

	if(!workCopy->histogramEnabled()) return;

	QPainter drawer(&viewHistogramBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TFPaintingPoint origin(inputArea_.x(), inputArea_.y());

	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		drawer.setPen(hist_);	
		drawer.drawLine(origin.x + i, origin.y + (1 - workCopy->getHistogramValue(i))*inputArea_.height(),
			origin.x + i + 1, origin.y + (1 - workCopy->getHistogramValue(i+1))*inputArea_.height());
	}
}

void TFRGBaPainter::updateRedView_(TFWorkCopy::Ptr workCopy){
		
	viewRedBuffer_ = QPixmap(area_.width(), area_.height());
	viewRedBuffer_.fill(noColor_);

	QPainter drawer(&viewRedBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TFPaintingPoint origin(inputArea_.x(), inputArea_.y());

	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		drawer.setPen(red_);
		drawer.drawLine(origin.x + i, origin.y + (1 - workCopy->getComponent1(i))*inputArea_.height(),
			origin.x + i + 1, origin.y + (1 - workCopy->getComponent1(i+1))*inputArea_.height());	
	}
}

void TFRGBaPainter::updateGreenView_(TFWorkCopy::Ptr workCopy){
		
	viewGreenBuffer_ = QPixmap(area_.width(), area_.height());
	viewGreenBuffer_.fill(noColor_);

	QPainter drawer(&viewGreenBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TFPaintingPoint origin(inputArea_.x(), inputArea_.y());

	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		drawer.setPen(green_);
		drawer.drawLine(origin.x + i, origin.y + (1 - workCopy->getComponent2(i))*inputArea_.height(),
			origin.x + i + 1, origin.y + (1 - workCopy->getComponent2(i+1))*inputArea_.height());	
	}
}

void TFRGBaPainter::updateBlueView_(TFWorkCopy::Ptr workCopy){
		
	viewBlueBuffer_ = QPixmap(area_.width(), area_.height());
	viewBlueBuffer_.fill(noColor_);

	QPainter drawer(&viewBlueBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TFPaintingPoint origin(inputArea_.x(), inputArea_.y());

	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		drawer.setPen(blue_);
		drawer.drawLine(origin.x + i, origin.y + (1 - workCopy->getComponent3(i))*inputArea_.height(),
			origin.x + i + 1, origin.y + (1 - workCopy->getComponent3(i+1))*inputArea_.height());	
	}
}

void TFRGBaPainter::updateAlphaView_(TFWorkCopy::Ptr workCopy){
		
	viewAlphaBuffer_ = QPixmap(area_.width(), area_.height());
	viewAlphaBuffer_.fill(noColor_);

	if(!drawAlpha_) return;

	QPainter drawer(&viewAlphaBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TFPaintingPoint origin(inputArea_.x(), inputArea_.y());

	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		drawer.setPen(alpha_);
		drawer.drawLine(origin.x + i, origin.y + (1 - workCopy->getAlpha(i))*inputArea_.height(),
			origin.x + i + 1, origin.y + (1 - workCopy->getAlpha(i+1))*inputArea_.height());
	}
}

void TFRGBaPainter::updateBottomColorBarView_(TFWorkCopy::Ptr workCopy){
		
	viewBottomColorBarBuffer_ = QPixmap(area_.width(), area_.height());
	viewBottomColorBarBuffer_.fill(noColor_);

	QPainter drawer(&viewBottomColorBarBuffer_);
	drawer.setClipRect(bottomBarArea_.x() + 1, bottomBarArea_.y() + 1,
		bottomBarArea_.width() + 1, bottomBarArea_.height());

	TFColor tfColor;
	QColor qColor;
	for(int i = 0; i < inputArea_.width(); ++i)
	{
		tfColor = workCopy->getColor(i);
		qColor.setRgbF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
		drawer.setPen(qColor);

		drawer.drawLine(bottomBarArea_.x() + i + 1, bottomBarArea_.y(),
			bottomBarArea_.x() + i + 1, bottomBarArea_.y() + bottomBarArea_.height() - 1);
	}
}

QPixmap TFRGBaPainter::getView(TFWorkCopy::Ptr workCopy){

	bool change = false;
	if(sizeChanged_)
	{
		updateBackground_();
		change = true;
	}
	if(sizeChanged_ || workCopy->histogramChanged())
	{
		updateHistogramView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->component1Changed())
	{
		updateRedView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->component2Changed())
	{
		updateGreenView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->component3Changed())
	{
		updateBlueView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->alphaChanged())
	{
		updateAlphaView_(workCopy);
		change = true;
	}
	if(change)
	{
		updateBottomColorBarView_(workCopy);

		viewBuffer_ = QPixmap(area_.width(), area_.height());
		viewBuffer_.fill(noColor_);
		QPainter drawer(&viewBuffer_);

		drawer.drawPixmap(0, 0, viewBackgroundBuffer_);
		drawer.drawPixmap(0, 0, viewHistogramBuffer_);
		drawer.drawPixmap(0, 0, viewBlueBuffer_);
		drawer.drawPixmap(0, 0, viewGreenBuffer_);
		drawer.drawPixmap(0, 0, viewRedBuffer_);
		drawer.drawPixmap(0, 0, viewAlphaBuffer_);
		drawer.drawPixmap(0, 0, viewBottomColorBarBuffer_);
	}

	sizeChanged_ = false;
	return viewBuffer_;
}

} // namespace GUI
} // namespace M4D
