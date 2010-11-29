#include "TFRGBHolder.h"

namespace M4D {
namespace GUI {

TFRGBHolder::TFRGBHolder(QWidget* window): TFAbstractHolder(window){

	type_ = TFHOLDER_RGB;
}

TFRGBHolder::~TFRGBHolder(){}

void TFRGBHolder::setUp(const TFSize& index){

	index_ = index;
	painter_.setUp(this);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}

void TFRGBHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFRGBHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFRGBHolder::resizePainter_(const QRect& rect){

	updateFunction_();

	painter_.resize(rect);
	
	updatePainter_();
}

TFAbstractFunction* TFRGBHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
