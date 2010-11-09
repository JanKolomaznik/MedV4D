#include "TFRGBHolder.h"

namespace M4D {
namespace GUI {

TFRGBHolder::TFRGBHolder(QWidget* window){

	setParent(window);
	type_ = TFTYPE_RGB;
}

TFRGBHolder::~TFRGBHolder(){}

void TFRGBHolder::setUp(QWidget *parent, const QRect rect){

	painter_.setUp(this, PAINTER_MARGIN);
	size_changed(rect);
	setParent(parent);
	show();
}

void TFRGBHolder::save_(QFile &file){	//TODO
	/*
	updateFunction_();

	 TFGrayscaleXmlWriter writer;
     writer.write(&file, function_);*/
	 //writer.writeTestData(&file);	//testing
}

bool TFRGBHolder::load_(QFile &file){	//TODO
	/*
	TFGrayscaleXmlREADER reader;

	bool error = false;

	reader.readTestData(&function_);	//testing
	//reader.read(&file, &function_, error);

	if (error || reader.error())
	{
		return false;
	}

	calculate_(function_.getFunction(), painter_.getView());
	*/
	return true;
}

void TFRGBHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getRedView(), function_.getRedFunction());
	calculate_(painter_.getGreenView(), function_.getGreenFunction());
	calculate_(painter_.getBlueView(), function_.getBlueFunction());
}

void TFRGBHolder::updatePainter_(const QRect& rect){

	painter_.resize(rect);
	
	calculate_(function_.getRedFunction(), painter_.getRedView());
	calculate_(function_.getGreenFunction(), painter_.getGreenView());
	calculate_(function_.getBlueFunction(), painter_.getBlueView());
}

TFAbstractFunction* TFRGBHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
