#include "TFHSVaHolder.h"

namespace M4D {
namespace GUI {

TFHSVaHolder::TFHSVaHolder(QWidget* window): TFAbstractHolder(window){

	type_ = TFHOLDER_HSVA;
}

TFHSVaHolder::~TFHSVaHolder(){}

void TFHSVaHolder::setUp(const TFSize& index){

	index_ = index;
	painter_.setUp(this);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}

void TFHSVaHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFHSVaHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFHSVaHolder::resizePainter_(const QRect& rect){

	updateFunction_();

	painter_.resize(rect);
	
	updatePainter_();
}

TFAbstractFunction* TFHSVaHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
