#include "TFGrayscaleAlphaPainter.h"

namespace M4D {
namespace GUI {

TFGrayscaleAlphaPainter::TFGrayscaleAlphaPainter():
	TFSimplePainter(Qt::white, Qt::white, Qt::white){

	componentNames_.push_back("Gray");
	componentNames_.push_back("Opacity");
}


TFGrayscaleAlphaPainter::TFGrayscaleAlphaPainter(const QColor& gray,
												 const QColor& alpha):
	TFSimplePainter(gray, gray, gray, alpha){

	componentNames_.push_back("Gray");
	componentNames_.push_back("Opacity");
}

TFGrayscaleAlphaPainter::~TFGrayscaleAlphaPainter(){}

QPixmap TFGrayscaleAlphaPainter::getView(TFWorkCopy::Ptr workCopy){

	bool change = false;
	if(sizeChanged_)
	{
		updateBackground_();
		updateHistogramView_(workCopy);
		updateComponent1View_(workCopy);
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
		drawer.drawPixmap(0, 0, viewComponent1Buffer_);
		drawer.drawPixmap(0, 0, viewAlphaBuffer_);
		drawer.drawPixmap(0, 0, viewBottomColorBarBuffer_);
	}

	sizeChanged_ = false;
	return viewBuffer_;
}

} // namespace GUI
} // namespace M4D
