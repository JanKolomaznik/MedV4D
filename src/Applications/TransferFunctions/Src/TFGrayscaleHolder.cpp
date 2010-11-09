#include "TFGrayscaleHolder.h"

namespace M4D {
namespace GUI {

TFGrayscaleHolder::TFGrayscaleHolder(QWidget* window){

	setParent(window);
	type_ = TFTYPE_GRAYSCALE;
}

TFGrayscaleHolder::~TFGrayscaleHolder(){}

void TFGrayscaleHolder::setUp(QWidget *parent, const QRect rect){

	painter_.setUp(this, PAINTER_MARGIN);
	size_changed(rect);
	setParent(parent);
	show();
}

void TFGrayscaleHolder::save_(QFile &file){

	updateFunction_();

	 TFGrayscaleXmlWriter writer;
     writer.write(&file, function_);
	 //writer.writeTestData(&file);	//testing
}

bool TFGrayscaleHolder::load_(QFile &file){

	TFGrayscaleXmlREADER reader;

	bool error = false;

	reader.readTestData(&function_);	//testing
	//reader.read(&file, &function_, error);

	if (error || reader.error())
	{
		return false;
	}

	calculate_(function_.getFunction(), painter_.getView());

	return true;
}

void TFGrayscaleHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getView(), function_.getFunction());
}

void TFGrayscaleHolder::updatePainter_(const QRect& rect){

	painter_.resize(rect);
	
	calculate_(function_.getFunction(), painter_.getView());
}

TFAbstractFunction* TFGrayscaleHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
