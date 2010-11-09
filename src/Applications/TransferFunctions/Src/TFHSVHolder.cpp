#include "TFHSVHolder.h"

#include <QtGui/QColor>

namespace M4D {
namespace GUI {

TFHSVHolder::TFHSVHolder(QWidget* window){

	setParent(window);
	type_ = TFTYPE_HSV;
}

TFHSVHolder::~TFHSVHolder(){}

void TFHSVHolder::setUp(QWidget *parent, const QRect rect){

	painter_.setUp(this, PAINTER_MARGIN);
	size_changed(rect);
	setParent(parent);
	show();
}

void TFHSVHolder::save_(QFile &file){	//TODO
	/*
	updateFunction_();

	 TFGrayscaleXmlWriter writer;
     writer.write(&file, function_);*/
	 //writer.writeTestData(&file);	//testing
}

bool TFHSVHolder::load_(QFile &file){	//TODO
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

void TFHSVHolder::updateFunction_(){

	if(!painter_.changed()) return;

	TFFunctionMapPtr hue = painter_.getRedView();
	TFFunctionMapPtr saturation = painter_.getGreenView();
	TFFunctionMapPtr value = painter_.getBlueView();

	TFSize painterRange = hue->size();

	TFFunctionMapPtr red(new TFFunctionMap(painterRange));
	TFFunctionMapPtr green(new TFFunctionMap(painterRange));
	TFFunctionMapPtr blue(new TFFunctionMap(painterRange));

	convert_(hue, saturation, value, red, green, blue, CONVERT_HSV_TO_RGB);

	calculate_(red, function_.getRedFunction());
	calculate_(green, function_.getGreenFunction());
	calculate_(blue, function_.getBlueFunction());
}

void TFHSVHolder::updatePainter_(const QRect& rect){

	painter_.resize(rect);
	
	TFFunctionMapPtr red = function_.getRedFunction();
	TFFunctionMapPtr green = function_.getGreenFunction();
	TFFunctionMapPtr blue = function_.getBlueFunction();

	TFSize painterRange = red->size();

	TFFunctionMapPtr hue(new TFFunctionMap(painterRange));
	TFFunctionMapPtr saturation(new TFFunctionMap(painterRange));
	TFFunctionMapPtr value(new TFFunctionMap(painterRange));

	convert_(red, green, blue, hue, saturation, value, CONVERT_RGB_TO_HSV);

	calculate_(hue, painter_.getRedView());
	calculate_(saturation, painter_.getGreenView());
	calculate_(value, painter_.getBlueView());
}

void TFHSVHolder::convert_(const TFFunctionMapPtr sourceComponent1,
							const TFFunctionMapPtr sourceComponent2,
							const TFFunctionMapPtr sourceComponent3,
							TFFunctionMapPtr outcomeComponent1,
							TFFunctionMapPtr outcomeComponent2,
							TFFunctionMapPtr outcomeComponent3,
							ConversionType type){

	TFFunctionMapIt src1 = sourceComponent1->begin();
	TFFunctionMapIt src2 = sourceComponent2->begin();
	TFFunctionMapIt src3 = sourceComponent3->begin();

	TFFunctionMapIt out1 = outcomeComponent1->begin();
	TFFunctionMapIt out2 = outcomeComponent2->begin();
	TFFunctionMapIt out3 = outcomeComponent3->begin();

	TFSize range = sourceComponent1->size();
	for(TFSize i = 0; i < range; ++i)
	{
		QColor srcColor;

		switch(type)
		{
			case CONVERT_HSV_TO_RGB:
			{
				srcColor.setHsvF(*src1, *src2, *src3);
				*out1 = srcColor.redF();
				*out2 = srcColor.greenF();
				*out3 = srcColor.blueF();
				break;
			}
			case CONVERT_RGB_TO_HSV:
			{
				srcColor.setRgbF(*src1, *src2, *src3);
				*out1 = srcColor.hueF();
				*out2 = srcColor.saturationF();
				*out3 = srcColor.valueF();
				break;
			}
		}
		++src1;
		++src2;
		++src3;
		++out1;
		++out2;
		++out3;
	}
}

TFAbstractFunction* TFHSVHolder::getFunction_(){

	return &function_;
}

} // namespace GUI
} // namespace M4D
