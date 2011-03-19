#include "TFRGBaFunction.h"

namespace M4D {
namespace GUI {

TFRGBaFunction::TFRGBaFunction(const TF::Size domain){

	colorMap_ = TF::ColorMapPtr(new TF::ColorMap(domain));
	domain_ = domain;
	clear();
}

TFRGBaFunction::TFRGBaFunction(TFRGBaFunction &function){

	operator=(function);
}

TFRGBaFunction::~TFRGBaFunction(){}

TF::Color TFRGBaFunction::getMappedRGBfColor(const TF::Size value){

	return (*colorMap_)[value];
}

TFAbstractFunction::Ptr TFRGBaFunction::clone(){

	return TFAbstractFunction::Ptr(new TFRGBaFunction(*this));
}

} // namespace GUI
} // namespace M4D
