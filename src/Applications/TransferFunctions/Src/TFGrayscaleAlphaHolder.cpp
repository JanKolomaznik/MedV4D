#include "TFGrayscaleAlphaHolder.h"

namespace M4D {
namespace GUI {

TFGrayscaleAlphaHolder::TFGrayscaleAlphaHolder(QWidget* window){

	setParent(window);
	type_ = TFHOLDER_GRAYSCALE_ALPHA;
}

TFGrayscaleAlphaHolder::~TFGrayscaleAlphaHolder(){}

void TFGrayscaleAlphaHolder::setUp(QWidget *parent, const QRect rect){

	painter_.setUp(this);
	size_changed(rect);
	setParent(parent);
	show();

	QObject::connect( &painter_, SIGNAL(FunctionChanged()), this, SLOT(on_use_clicked()));
}

void TFGrayscaleAlphaHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getColorMap());
}

void TFGrayscaleAlphaHolder::updatePainter_(){
	
	calculate_(function_.getColorMap(), painter_.getView());
}

void TFGrayscaleAlphaHolder::resizePainter_(const QRect& rect){

	TFColorMapPtr oldView = painter_.getView();

	painter_.resize(rect);
	
	calculate_(oldView, painter_.getView());
}

TFAbstractFunction* TFGrayscaleAlphaHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
