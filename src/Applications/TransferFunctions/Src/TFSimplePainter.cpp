#include "TFSimplePainter.h"

namespace M4D {
namespace GUI {

TFSimplePainter::TFSimplePainter(const QColor& component1):
	margin_(5),
	spacing_(5),
	colorBarSize_(10),
	background_(Qt::black),
	component1_(component1),
	hist_(Qt::darkGray),
	noColor_(0,0,0,0),
	drawAlpha_(false),
	firstOnly_(true),
	sizeChanged_(true){
}

TFSimplePainter::TFSimplePainter(const QColor& component1,
								 const QColor& alpha):
	margin_(5),
	spacing_(5),
	colorBarSize_(10),
	background_(Qt::black),
	component1_(component1),
	alpha_(alpha),
	hist_(Qt::darkGray),
	noColor_(0,0,0,0),
	drawAlpha_(true),
	firstOnly_(true),
	sizeChanged_(true){
}

TFSimplePainter::TFSimplePainter(const QColor& component1,
								 const QColor& component2,
								 const QColor& component3):
	margin_(5),
	spacing_(5),
	colorBarSize_(10),
	background_(Qt::black),
	component1_(component1),
	component2_(component2),
	component3_(component3),
	hist_(Qt::darkGray),
	noColor_(0,0,0,0),
	drawAlpha_(false),
	firstOnly_(false),
	sizeChanged_(true){
}

TFSimplePainter::TFSimplePainter(const QColor& component1,
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
	drawAlpha_(true),
	firstOnly_(false),
	sizeChanged_(true){
}

TFSimplePainter::~TFSimplePainter(){}

void TFSimplePainter::setArea(QRect area){
	
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

QRect TFSimplePainter::getInputArea(){

	return inputArea_;
}

void TFSimplePainter::updateBackground_(){

	viewBackgroundBuffer_ = QPixmap(area_.width(), area_.height());
	viewBackgroundBuffer_.fill(noColor_);

	QPainter drawer(&viewBackgroundBuffer_);
	drawer.fillRect(backgroundArea_, QBrush(background_));
	drawer.fillRect(bottomBarArea_, QBrush(background_));
}

void TFSimplePainter::updateHistogramView_(WorkCopy::Ptr workCopy){
		
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

void TFSimplePainter::updateComponent1View_(WorkCopy::Ptr workCopy){
		
	viewComponent1Buffer_ = QPixmap(area_.width(), area_.height());
	viewComponent1Buffer_.fill(noColor_);

	QPainter drawer(&viewComponent1Buffer_);
	drawer.setClipRect(inputArea_.x(), inputArea_.y(),
		inputArea_.width() + 1, inputArea_.height() + 1);

	TF::PaintingPoint origin(inputArea_.x(), inputArea_.y());

	int x1, y1, x2, y2;
	for(int i = 0; i < inputArea_.width() - 1; ++i)
	{
		x1 = origin.x + i;
		y1 = origin.y + (1 - workCopy->getComponent1(i, TF_DIMENSION_1))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent1(i + 1, TF_DIMENSION_1))*inputArea_.height();

		drawer.setPen(component1_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFSimplePainter::updateComponent2View_(WorkCopy::Ptr workCopy){
		
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
		y1 = origin.y + (1 - workCopy->getComponent2(i, TF_DIMENSION_1))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent2(i + 1, TF_DIMENSION_1))*inputArea_.height();

		drawer.setPen(component2_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFSimplePainter::updateComponent3View_(WorkCopy::Ptr workCopy){
		
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
		y1 = origin.y + (1 - workCopy->getComponent3(i, TF_DIMENSION_1))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getComponent3(i + 1, TF_DIMENSION_1))*inputArea_.height();

		drawer.setPen(component3_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFSimplePainter::updateAlphaView_(WorkCopy::Ptr workCopy){
		
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
		y1 = origin.y + (1 - workCopy->getAlpha(i, TF_DIMENSION_1))*inputArea_.height();
		x2 = origin.x + i + 1;
		y2 = origin.y + (1 - workCopy->getAlpha(i + 1, TF_DIMENSION_1))*inputArea_.height();

		drawer.setPen(alpha_);
		drawer.drawLine(x1, y1,	x2, y2);	
	}
}

void TFSimplePainter::updateBottomColorBarView_(WorkCopy::Ptr workCopy){
		
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
		tfColor = workCopy->getColor(i, TF_DIMENSION_1);

		qColor.setRgbF(tfColor.component1, tfColor.component2, tfColor.component3, tfColor.alpha);
		drawer.setPen(qColor);
		
		x1 = bottomBarArea_.x() + i + 1;
		y1 = bottomBarArea_.y();
		x2 = bottomBarArea_.x() + i + 1;
		y2 = bottomBarArea_.y() + bottomBarArea_.height() - 1;

		drawer.drawLine(x1, y1, x2, y2);
	}
}

QPixmap TFSimplePainter::getView(WorkCopy::Ptr workCopy){

	bool change = false;
	if(sizeChanged_)
	{
		updateBackground_();
		updateHistogramView_(workCopy);
		updateComponent1View_(workCopy);
		if(!firstOnly_)
		{
			updateComponent2View_(workCopy);
			updateComponent3View_(workCopy);
		}
		updateAlphaView_(workCopy);
		change = true;
	}
	else
	{
		if(workCopy->histogramChanged())
		{
			updateHistogramView_(workCopy);
			change = true;
		}
		if(workCopy->component1Changed(TF_DIMENSION_1))
		{
			updateComponent1View_(workCopy);
			change = true;
		}
		if(!firstOnly_)
		{
			if(workCopy->component2Changed(TF_DIMENSION_1))
			{
				updateComponent2View_(workCopy);
				change = true;
			}
			if(workCopy->component3Changed(TF_DIMENSION_1))
			{
				updateComponent3View_(workCopy);
				change = true;
			}
		}
		if(workCopy->alphaChanged(TF_DIMENSION_1))
		{
			updateAlphaView_(workCopy);
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
		drawer.drawPixmap(0, 0, viewComponent3Buffer_);
		drawer.drawPixmap(0, 0, viewComponent2Buffer_);
		drawer.drawPixmap(0, 0, viewComponent1Buffer_);
		drawer.drawPixmap(0, 0, viewAlphaBuffer_);
		drawer.drawPixmap(0, 0, viewBottomColorBarBuffer_);
	}

	sizeChanged_ = false;
	return viewBuffer_;
}

} // namespace GUI
} // namespace M4D
