#include "TFHSVaFunction.h"

namespace M4D {
namespace GUI {

TFHSVaFunction::TFHSVaFunction(const TF::Size domain){

	colorMap_ = TF::ColorMapPtr(new TF::ColorMap(domain));
	domain_ = domain;
	clear();
}

TFHSVaFunction::TFHSVaFunction(TFHSVaFunction &function){

	operator=(function);
}

TFHSVaFunction::~TFHSVaFunction(){}

TF::Color TFHSVaFunction::getMappedRGBfColor(const TF::Size value){

	QColor color;
	color.setHsvF(
		(*colorMap_)[value].component1,
		(*colorMap_)[value].component2,
		(*colorMap_)[value].component3,
		(*colorMap_)[value].alpha);
	
	return TF::Color(color.redF(), color.greenF(), color.blueF(), color.alphaF());
}

TFAbstractFunction::Ptr TFHSVaFunction::clone(){

	return TFAbstractFunction::Ptr(new TFHSVaFunction(*this));
}

} // namespace GUI
} // namespace M4D
