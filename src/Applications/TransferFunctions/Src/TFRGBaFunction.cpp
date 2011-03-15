#include "TFRGBaFunction.h"

namespace M4D {
namespace GUI {

TFRGBaFunction::TFRGBaFunction(const TFSize domain){

	type_ = TFFUNCTION_RGBA;
	colorMap_ = TFColorMapPtr(new TFColorMap(domain));
	domain_ = domain;
	clear();
}

TFRGBaFunction::TFRGBaFunction(TFRGBaFunction &function){

	operator=(function);
}

TFRGBaFunction::~TFRGBaFunction(){}

TFColor TFRGBaFunction::getMappedRGBfColor(const TFSize value){

	return (*colorMap_)[value];
}

} // namespace GUI
} // namespace M4D
