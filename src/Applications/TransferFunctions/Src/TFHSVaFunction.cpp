#include "TFHSVaFunction.h"

namespace M4D {
namespace GUI {

TFHSVaFunction::TFHSVaFunction(const TFSize domain){

	type_ = TFFUNCTION_HSVA;
	colorMap_ = TFColorMapPtr(new TFColorMap(domain));
	domain_ = domain;
	clear();
}

TFHSVaFunction::TFHSVaFunction(TFHSVaFunction &function){

	operator=(function);
}

TFHSVaFunction::~TFHSVaFunction(){}

TFColor TFHSVaFunction::getMappedRGBfColor(const TFSize value){

	QColor color;
	color.setHsvF(
		(*colorMap_)[value].component1,
		(*colorMap_)[value].component2,
		(*colorMap_)[value].component3,
		(*colorMap_)[value].alpha);
	
	return TFColor(color.redF(), color.greenF(), color.blueF(), color.alphaF());
}

} // namespace GUI
} // namespace M4D
