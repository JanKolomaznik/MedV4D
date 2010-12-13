#include "TFHSVHolder.h"

namespace M4D {
namespace GUI {

TFHSVHolder::TFHSVHolder(QMainWindow* parent):
	TFAbstractHolder(parent),
	painter_(basicTools_->painterWidget){

	type_ = TFHOLDER_HSV;

	painter_.correctView();
}

TFHSVHolder::~TFHSVHolder(){
}
/*
void TFHSVHolder::setUp(TFSize index){

	index_ = index;
	painter_.setUp(basicTools_->painterWidget);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}
*/
void TFHSVHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFHSVHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFHSVHolder::resizePainter_(){

	updateFunction_();

	painter_.resize(basicTools_->painterWidget->size());
	painter_.correctView();
	
	updatePainter_();
}

TFAbstractFunction* TFHSVHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
