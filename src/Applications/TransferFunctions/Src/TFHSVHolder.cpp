#include "TFHSVHolder.h"

namespace M4D {
namespace GUI {

TFHSVHolder::TFHSVHolder(QWidget* window): TFAbstractHolder(window){

	type_ = TFHOLDER_HSV;
}

TFHSVHolder::~TFHSVHolder(){}

void TFHSVHolder::setUp(const TFSize& index){

	index_ = index;
	painter_.setUp(this);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}

void TFHSVHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFHSVHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFHSVHolder::resizePainter_(const QRect& rect){

	updateFunction_();

	painter_.resize(rect);
	
	updatePainter_();
}

TFAbstractFunction* TFHSVHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
