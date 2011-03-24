#include "TFGrayscaleAlphaPainter.h"

namespace M4D {
namespace GUI {

TFGrayscaleAlphaPainter::TFGrayscaleAlphaPainter(bool drawAlpha):
	margin_(5),
	spacing_(5),
	colorBarSize_(10),
	background_(Qt::black),
	gray_(Qt::lightGray),
	alpha_(Qt::yellow),
	hist_(255,140,0,255),
	noColor_(0,0,0,0),
	drawAlpha_(drawAlpha){
}

TFGrayscaleAlphaPainter::~TFGrayscaleAlphaPainter(){}

void TFGrayscaleAlphaPainter::setArea(QRect area){
	
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

QRect TFGrayscaleAlphaPainter::getInputArea(){

	return QRect(area_.x() + inputArea_.x(), area_.y() + inputArea_.y(),
		inputArea_.width(), inputArea_.height());
}

void TFGrayscaleAlphaPainter::updateBackground_(){

	viewBackgroundBuffer_ = QPixmap(area_.width(), area_.height());
	viewBackgroundBuffer_.fill(noColor_);

	QPainter drawer(&viewBackgroundBuffer_);
	drawer.fillRect(backgroundArea_, QBrush(background_));
	drawer.fillRect(bottomBarArea_, QBrush(background_));
}

void TFGrayscaleAlphaPainter::updateHistogramView_(TFWorkCopy<TF_GRAYSCALEPAINTER_DIMENSION>::Ptr workCopy){
		
	viewHistogramBuffer_ = QPixmap(area_.width(), area_.height());
	viewHistogramBuffer_.fill(noColor_);

	if(!workCopy->histogramEnabled()) return;

	QPainter drawer(&viewHistogramBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getHistogramValue(i))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getHistogramValue(i + 1))*inputArea_.height();

		drawer.setPen(hist_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFGrayscaleAlphaPainter::updateGrayView_(TFWorkCopy<TF_GRAYSCALEPAINTER_DIMENSION>::Ptr workCopy){
		
	viewGrayBuffer_ = QPixmap(area_.width(), area_.height());
	viewGrayBuffer_.fill(noColor_);

	QPainter drawer(&viewGrayBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent1(i, TF_GRAYSCALEPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent1(i + 1, TF_GRAYSCALEPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(gray_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFGrayscaleAlphaPainter::updateAlphaView_(TFWorkCopy<TF_GRAYSCALEPAINTER_DIMENSION>::Ptr workCopy){
		
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
		y1 = origin.y + (1 - workCopy->getAlpha(i, TF_GRAYSCALEPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getAlpha(i + 1, TF_GRAYSCALEPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(alpha_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFGrayscaleAlphaPainter::updateBottomColorBarView_(TFWorkCopy<TF_GRAYSCALEPAINTER_DIMENSION>::Ptr workCopy){
		
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
		tfColor = workCopy->getColor(i, TF_GRAYSCALEPAINTER_DIMENSION);
		qColor.setRgbF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
		drawer.setPen(qColor);
		
		x1 = bottomBarArea_.x() + i + 1;
		y1 = bottomBarArea_.y();
		x2 = bottomBarArea_.x() + i + 1;
		y2 = bottomBarArea_.y() + bottomBarArea_.height() - 1;

		drawer.drawLine(x1, y1, x2, y2);
	}
}

QPixmap TFGrayscaleAlphaPainter::getView(TFWorkCopy<TF_GRAYSCALEPAINTER_DIMENSION>::Ptr workCopy){

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
	if(sizeChanged_ || workCopy->component1Changed(TF_GRAYSCALEPAINTER_DIMENSION))
	{
		updateGrayView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->alphaChanged(TF_GRAYSCALEPAINTER_DIMENSION))
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
		drawer.drawPixmap(0, 0, viewGrayBuffer_);
		drawer.drawPixmap(0, 0, viewAlphaBuffer_);
		drawer.drawPixmap(0, 0, viewBottomColorBarBuffer_);
	}

	sizeChanged_ = false;
	return viewBuffer_;
}

} // namespace GUI
} // namespace M4D
