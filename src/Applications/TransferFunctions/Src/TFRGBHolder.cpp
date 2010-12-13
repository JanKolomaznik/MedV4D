#include "TFRGBHolder.h"

namespace M4D {
namespace GUI {

TFRGBHolder::TFRGBHolder(QMainWindow* parent):
	TFAbstractHolder(parent),
	painter_(basicTools_->painterWidget){

	type_ = TFHOLDER_RGB;

	painter_.correctView();
}

TFRGBHolder::~TFRGBHolder(){
}
/*
void TFRGBHolder::setUp(TFSize index){

	index_ = index;
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}
*/
void TFRGBHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFRGBHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFRGBHolder::resizePainter_(){

	updateFunction_();

	painter_.resize(basicTools_->painterWidget->size());
	painter_.correctView();
	
	updatePainter_();
}

TFAbstractFunction* TFRGBHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
