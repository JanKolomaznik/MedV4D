#include "TFRGBaHolder.h"

namespace M4D {
namespace GUI {

TFRGBaHolder::TFRGBaHolder(QWidget* window): TFAbstractHolder(window){

	type_ = TFHOLDER_RGBA;
}

TFRGBaHolder::~TFRGBaHolder(){}

void TFRGBaHolder::setUp(const TFSize& index){

	index_ = index;
	painter_.setUp(this);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}

void TFRGBaHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFRGBaHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFRGBaHolder::resizePainter_(const QRect& rect){

	updateFunction_();

	painter_.resize(rect);
	
	updatePainter_();
}

TFAbstractFunction* TFRGBaHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
