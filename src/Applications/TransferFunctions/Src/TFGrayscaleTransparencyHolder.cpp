#include "TFGrayscaleTransparencyHolder.h"

namespace M4D {
namespace GUI {

TFGrayscaleTransparencyHolder::TFGrayscaleTransparencyHolder(QWidget* window){

	setParent(window);
	type_ = TFTYPE_GRAYSCALE_TRANSPARENCY;
}

TFGrayscaleTransparencyHolder::~TFGrayscaleTransparencyHolder(){}

void TFGrayscaleTransparencyHolder::setUp(QWidget *parent, const QRect rect){

	painter_.setUp(this, PAINTER_MARGIN);
	size_changed(rect);
	setParent(parent);
	show();

	QObject::connect( &painter_, SIGNAL(FunctionChanged()), this, SLOT(on_use_clicked()));
}

void TFGrayscaleTransparencyHolder::save_(QFile &file){	//TODO
	/*
	updateFunction_();

	 TFXmlSimpleWriter writer;
     writer.write(&file, function_);*/
	 //writer.writeTestData(&file);	//testing
}

bool TFGrayscaleTransparencyHolder::load_(QFile &file){	//TODO
	/*
	TFXmlSimpleReader reader;

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

void TFGrayscaleTransparencyHolder::updateFunction_(){

	if(!painter_.changed()) return;

	calculate_(painter_.getGrayscaleView(), function_.getGrayscaleFunction());
	calculate_(painter_.getTransparencyView(), function_.getTransparencyFunction());
}

void TFGrayscaleTransparencyHolder::size_changed(const QRect rect){
	
	setGeometry(rect);

	int newWidth = rect.width() - 2*PAINTER_X;
	int newHeight = rect.height() - 2*PAINTER_Y;

	updateFunction_();

	painter_.resize(QRect(PAINTER_X, PAINTER_Y, newWidth, newHeight));

	calculate_(function_.getGrayscaleFunction(), painter_.getGrayscaleView());
	calculate_(function_.getTransparencyFunction(), painter_.getTransparencyView());
}

TFAbstractFunction* TFGrayscaleTransparencyHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
