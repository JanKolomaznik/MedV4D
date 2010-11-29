#include "TFGrayscaleHolder.h"

namespace M4D {
namespace GUI {

TFGrayscaleHolder::TFGrayscaleHolder(QWidget* window): TFAbstractHolder(window){

	type_ = TFHOLDER_GRAYSCALE;
}

TFGrayscaleHolder::~TFGrayscaleHolder(){}

void TFGrayscaleHolder::setUp(const TFSize& index){

	index_ = index;
	painter_.setUp(this);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}

void TFGrayscaleHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFGrayscaleHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFGrayscaleHolder::resizePainter_(const QRect& rect){

	updateFunction_();

	painter_.resize(rect);
	
	updatePainter_();
}

TFAbstractFunction* TFGrayscaleHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
