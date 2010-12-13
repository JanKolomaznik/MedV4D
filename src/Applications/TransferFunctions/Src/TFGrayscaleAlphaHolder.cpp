#include "TFGrayscaleAlphaHolder.h"

namespace M4D {
namespace GUI {

TFGrayscaleAlphaHolder::TFGrayscaleAlphaHolder(QMainWindow* parent):
	TFAbstractHolder(parent),
	painter_(basicTools_->painterWidget){

	type_ = TFHOLDER_GRAYSCALE_ALPHA;

	painter_.correctView();
}

TFGrayscaleAlphaHolder::~TFGrayscaleAlphaHolder(){
}
/*
void TFGrayscaleAlphaHolder::setUp(TFSize index){

	index_ = index;
	painter_.setUp(basicTools_->painterWidget);
	size_changed(index_, rect());
	setWindowTitle(convert);
	show();
}
*/
void TFGrayscaleAlphaHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFGrayscaleAlphaHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFGrayscaleAlphaHolder::resizePainter_(){

	updateFunction_();

	painter_.resize(basicTools_->painterWidget->size());
	painter_.correctView();
	
	updatePainter_();
}

TFAbstractFunction* TFGrayscaleAlphaHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
