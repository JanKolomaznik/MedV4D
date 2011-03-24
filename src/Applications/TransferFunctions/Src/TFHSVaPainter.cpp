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
	hist_(255,140,0,255),
	noColor_(0,0,0,0),
	drawAlpha_(drawAlpha),
	sizeChanged_(true){
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

	sizeChanged_ = true;
}

QRect TFHSVaPainter::getInputArea(){

	return QRect(area_.x() + inputArea_.x(), area_.y() + inputArea_.y(),
		inputArea_.width(), inputArea_.height());
}

void TFHSVaPainter::updateBackground_(){

	viewBackgroundBuffer_ = QPixmap(area_.width(), area_.height());
	viewBackgroundBuffer_.fill(noColor_);

	QPainter drawer(&viewBackgroundBuffer_);
	drawer.fillRect(backgroundArea_, QBrush(background_));
	drawer.fillRect(bottomBarArea_, QBrush(background_));

	QColor color;
	int x1, y1, x2, y2;
	for(int i = 0; i < sideBarArea_.height() - 1; ++i)
	{
		x1 = sideBarArea_.x();
		y1 = sideBarArea_.y() + sideBarArea_.height() - i;
		x2 = sideBarArea_.x() + sideBarArea_.width();
		y2 = sideBarArea_.y() + sideBarArea_.height() - i;

		color.setHsvF(i/(float)sideBarArea_.height(), 1, 1);

		drawer.setPen(color);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFHSVaPainter::updateHistogramView_(TFWorkCopy<TF_HSVPAINTER_DIMENSION>::Ptr workCopy){
		
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

void TFHSVaPainter::updateHueView_(TFWorkCopy<TF_HSVPAINTER_DIMENSION>::Ptr workCopy){
		
	viewHueBuffer_ = QPixmap(area_.width(), area_.height());
	viewHueBuffer_.fill(noColor_);

	QPainter drawer(&viewHueBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent1(i, TF_HSVPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent1(i + 1, TF_HSVPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(hue_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFHSVaPainter::updateSaturationView_(TFWorkCopy<TF_HSVPAINTER_DIMENSION>::Ptr workCopy){
		
	viewSaturationBuffer_ = QPixmap(area_.width(), area_.height());
	viewSaturationBuffer_.fill(noColor_);

	QPainter drawer(&viewSaturationBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent2(i, TF_HSVPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent2(i + 1, TF_HSVPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(saturation_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFHSVaPainter::updateValueView_(TFWorkCopy<TF_HSVPAINTER_DIMENSION>::Ptr workCopy){
		
	viewValueBuffer_ = QPixmap(area_.width(), area_.height());
	viewValueBuffer_.fill(noColor_);

	QPainter drawer(&viewValueBuffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent3(i, TF_HSVPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent3(i + 1, TF_HSVPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(value_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFHSVaPainter::updateAlphaView_(TFWorkCopy<TF_HSVPAINTER_DIMENSION>::Ptr workCopy){
		
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
		y1 = origin.y + (1 - workCopy->getAlpha(i, TF_HSVPAINTER_DIMENSION))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getAlpha(i + 1, TF_HSVPAINTER_DIMENSION))*inputArea_.height();

		drawer.setPen(alpha_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFHSVaPainter::updateBottomColorBarView_(TFWorkCopy<TF_HSVPAINTER_DIMENSION>::Ptr workCopy){
		
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
		tfColor = workCopy->getColor(i, TF_HSVPAINTER_DIMENSION);
		qColor.setRgbF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
		drawer.setPen(qColor);
		
		x1 = bottomBarArea_.x() + i + 1;
		y1 = bottomBarArea_.y();
		x2 = bottomBarArea_.x() + i + 1;
		y2 = bottomBarArea_.y() + bottomBarArea_.height() - 1;

		drawer.drawLine(x1, y1, x2, y2);
	}
}

QPixmap TFHSVaPainter::getView(TFWorkCopy<TF_HSVPAINTER_DIMENSION>::Ptr workCopy){

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
	if(sizeChanged_ || workCopy->component1Changed(TF_HSVPAINTER_DIMENSION))
	{
		updateHueView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->component2Changed(TF_HSVPAINTER_DIMENSION))
	{
		updateSaturationView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->component3Changed(TF_HSVPAINTER_DIMENSION))
	{
		updateValueView_(workCopy);
		change = true;
	}
	if(sizeChanged_ || workCopy->alphaChanged(TF_HSVPAINTER_DIMENSION))
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
		drawer.drawPixmap(0, 0, viewValueBuffer_);
		drawer.drawPixmap(0, 0, viewSaturationBuffer_);
		drawer.drawPixmap(0, 0, viewHueBuffer_);
		drawer.drawPixmap(0, 0, viewAlphaBuffer_);
		drawer.drawPixmap(0, 0, viewBottomColorBarBuffer_);
	}

	sizeChanged_ = false;
	return viewBuffer_;
}

} // namespace GUI
} // namespace M4D
