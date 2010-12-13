#include "TFHSVaHolder.h"

namespace M4D {
namespace GUI {

TFHSVaHolder::TFHSVaHolder(QMainWindow* parent):
	TFAbstractHolder(parent),
	painter_(basicTools_->painterWidget){

	type_ = TFHOLDER_HSVA;

	painter_.correctView();
}

TFHSVaHolder::~TFHSVaHolder(){
}
/*
void TFHSVaHolder::setUp(TFSize index){

	index_ = index;
	painter_.setUp(basicTools_->painterWidget);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}
*/
void TFHSVaHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFHSVaHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFHSVaHolder::resizePainter_(){

	updateFunction_();

	painter_.resize(basicTools_->painterWidget->size());
	painter_.correctView();
	
	updatePainter_();
}

TFAbstractFunction* TFHSVaHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
