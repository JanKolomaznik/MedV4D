#include "TFRGBaFunction.h"

namespace M4D {
namespace GUI {

TFRGBaFunction::TFRGBaFunction(TFSize domain){

	type_ = TFFUNCTION_RGBA;
	colorMap_ = TFColorMapPtr(new TFColorMap(domain));
	clear();
}

TFRGBaFunction::TFRGBaFunction(TFRGBaFunction &function){

	operator=(function);
}

TFRGBaFunction::~TFRGBaFunction(){}

} // namespace GUI
} // namespace M4D
