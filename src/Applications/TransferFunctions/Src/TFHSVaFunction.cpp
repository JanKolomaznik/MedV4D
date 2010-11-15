#include "TFHSVaFunction.h"

namespace M4D {
namespace GUI {

TFHSVaFunction::TFHSVaFunction(TFSize domain){

	type_ = TFFUNCTION_HSVA;
	colorMap_ = TFColorMapPtr(new TFColorMap(domain));
	clear();
}

TFHSVaFunction::TFHSVaFunction(TFHSVaFunction &function){

	operator=(function);
}

TFHSVaFunction::~TFHSVaFunction(){}

} // namespace GUI
} // namespace M4D
