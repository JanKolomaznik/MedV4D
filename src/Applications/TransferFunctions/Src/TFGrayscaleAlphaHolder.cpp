#include "TFGrayscaleAlphaHolder.h"

namespace M4D {
namespace GUI {

TFGrayscaleAlphaHolder::TFGrayscaleAlphaHolder(QWidget* window): TFAbstractHolder(window){

	type_ = TFHOLDER_GRAYSCALE_ALPHA;
}

TFGrayscaleAlphaHolder::~TFGrayscaleAlphaHolder(){}

void TFGrayscaleAlphaHolder::setUp(const TFSize& index){

	index_ = index;
	painter_.setUp(this);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}

void TFGrayscaleAlphaHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFGrayscaleAlphaHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFGrayscaleAlphaHolder::resizePainter_(const QRect& rect){

	updateFunction_();

	painter_.resize(rect);
	
	updatePainter_();
}

TFAbstractFunction* TFGrayscaleAlphaHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
