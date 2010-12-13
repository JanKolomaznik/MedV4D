#include "TFGrayscaleHolder.h"

namespace M4D {
namespace GUI {

TFGrayscaleHolder::TFGrayscaleHolder(QMainWindow* parent):
	TFAbstractHolder(parent),
	painter_(basicTools_->painterWidget){

	type_ = TFHOLDER_GRAYSCALE;

	painter_.correctView();
}

TFGrayscaleHolder::~TFGrayscaleHolder(){
}
/*
void TFGrayscaleHolder::setUp(TFSize index){

	index_ = index;
	painter_.setUp(basicTools_->painterWidget);
	size_changed(index_, dynamic_cast<QWidget*>(parent())->rect());
	show();
}
*/
void TFGrayscaleHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFGrayscaleHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFGrayscaleHolder::resizePainter_(){

	updateFunction_();

	painter_.resize(basicTools_->painterWidget->size());
	painter_.correctView();
	
	updatePainter_();
}

TFAbstractFunction* TFGrayscaleHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
