#include "TFGrayscaleHolder.h"

namespace M4D {
namespace GUI {

TFGrayscaleHolder::TFGrayscaleHolder(QWidget* window){

	setParent(window);
	type_ = TFHOLDER_GRAYSCALE;
}

TFGrayscaleHolder::~TFGrayscaleHolder(){}

void TFGrayscaleHolder::setUp(QWidget *parent, const QRect& rect){

	painter_.setUp(this);
	size_changed(rect);
	setParent(parent);
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
