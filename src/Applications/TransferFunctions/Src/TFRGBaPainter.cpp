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
	hist_(Qt::darkGray),
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

	return QRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width(), inputArea_.height());
}

void TFRGBaPainter::updateBackground_(){

	viewBackgroundBuffer_ = QPixmap(area_.width(), area_.height());
	viewBackgroundBuffer_.fill(noColor_);

	QPainter drawer(&viewBackgroundBuffer_);
	drawer.fillRect(backgroundArea_, QBrush(background_));
	drawer.fillRect(bottomBarArea_, QBrush(background_));
}

void TFRGBaPainter::updateHistogramView_(WorkCopy::Ptr workCopy){
		
	viewHistogramBuffer_ = QPixmap(area_.width(), area_.height());
	viewHistogramBuffer_.fill(noColor_);

	if(!workCopy->histogramEnabled()) return;

	QPainter drawer(&viewHistogramBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2 = origin.y + inputArea_.height();
	for(int i = 0; i < inputArea_.width(); ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getHistogramValue(i))*inputArea_.height();
		x2 = origin.x + i;

		drawer.setPen(hist_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFRGBaPainter::updateRedView_(WorkCopy::Ptr workCopy){
		
	viewRedBuffer_ = QPixmap(area_.width(), area_.height());
	viewRedBuffer_.fill(noColor_);

	QPainter drawer(&viewRedBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent1(i, TF_RGBPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent1(i + 1, TF_RGBPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(red_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFRGBaPainter::updateGreenView_(WorkCopy::Ptr workCopy){
		
	viewGreenBuffer_ = QPixmap(area_.width(), area_.height());
	viewGreenBuffer_.fill(noColor_);

	QPainter drawer(&viewGreenBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent2(i, TF_RGBPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent2(i + 1, TF_RGBPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(green_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFRGBaPainter::updateBlueView_(WorkCopy::Ptr workCopy){
		
	viewBlueBuffer_ = QPixmap(area_.width(), area_.height());
	viewBlueBuffer_.fill(noColor_);

	QPainter drawer(&viewBlueBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent3(i, TF_RGBPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent3(i + 1, TF_RGBPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(blue_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFRGBaPainter::updateAlphaView_(WorkCopy::Ptr workCopy){
		
	viewAlphaBuffer_ = QPixmap(area_.width(), area_.height());
	viewAlphaBuffer_.fill(noColor_);

	if(!drawAlpha_) return;

	QPainter drawer(&viewAlphaBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getAlpha(i, TF_RGBPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getAlpha(i + 1, TF_RGBPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(alpha_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFRGBaPainter::updateBottomColorBarView_(WorkCopy::Ptr workCopy){
		
	viewBottomColorBarBuffer_ = QPixmap(area_.width(), area_.height());
	viewBottomColorBarBuffer_.fill(noColor_);

	QPainter drawer(&viewBottomColorBarBuffer_);
	drawer.setClipRect(bottomBarArea_.x() + 1, bottomBarArea_.y() + 1,
		bottomBarArea_.width() + 1, bottomBarArea_.height());

	TF::Color tfColor;
	QColor qColor;
	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width(); ++i)
	{
		tfColor = workCopy->getColor(i, TF_RGBPAINTER_DIMENSION);

		qColor.setRgbF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
		drawer.setPen(qColor);
		
		x1 = bottomBarArea_.x() + i + 1;
		y1 = bottomBarArea_.y();
		x2 = bottomBarArea_.x() + i + 1;
		y2 = bottomBarArea_.y() + bottomBarArea_.height() - 1;

		drawer.drawLine(x1, y1, x2, y2);
	}
}

QPixmap TFRGBaPainter::getView(WorkCopy::Ptr workCopy){

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
	if(sizeChanged_ || workCopy->component1Changed(TF_RGBPAINTER_DIMENSION))
	{
		updateRedView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->component2Changed(TF_RGBPAINTER_DIMENSION))
	{
		updateGreenView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->component3Changed(TF_RGBPAINTER_DIMENSION))
	{
		updateBlueView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->alphaChanged(TF_RGBPAINTER_DIMENSION))
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
