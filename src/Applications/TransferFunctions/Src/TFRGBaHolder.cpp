#include "TFRGBaHolder.h"

namespace M4D {
namespace GUI {

TFRGBaHolder::TFRGBaHolder(QMainWindow* parent):
	TFAbstractHolder(parent),
	painter_(basicTools_->painterWidget){

	type_ = TFHOLDER_RGBA;

	painter_.correctView();
}

TFRGBaHolder::~TFRGBaHolder(){
}
/*
void TFRGBaHolder::setUp(TFSize index){

	index_ = index;
	painter_.setUp(basicTools_->painterWidget);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}
*/
void TFRGBaHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFRGBaHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFRGBaHolder::resizePainter_(){

	updateFunction_();

	painter_.resize(basicTools_->painterWidget->size());
	painter_.correctView();
	
	updatePainter_();
}

TFAbstractFunction* TFRGBaHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
