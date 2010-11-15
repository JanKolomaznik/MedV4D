#include "TFRGBHolder.h"

namespace M4D {
namespace GUI {

TFRGBHolder::TFRGBHolder(QWidget* window){

	setParent(window);
	type_ = TFHOLDER_RGB;
}

TFRGBHolder::~TFRGBHolder(){}

void TFRGBHolder::setUp(QWidget *parent, const QRect rect){

	painter_.setUp(this);
	size_changed(rect);
	setParent(parent);
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

	TFColorMapPtr oldView = painter_.getView();

	painter_.resize(rect);
	
	calculate_(oldView, painter_.getView());
}

TFAbstractFunction* TFRGBHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
