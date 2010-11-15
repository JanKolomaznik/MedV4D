#include "TFRGBFunction.h"

namespace M4D {
namespace GUI {

TFRGBFunction::TFRGBFunction(TFSize domain){

	type_ = TFTYPE_RGB;
	red_ = TFColorRGBaMapPtr(new TFColorRGBaMap(domain));
	green_ = TFColorRGBaMapPtr(new TFColorRGBaMap(domain));
	blue_ = TFColorRGBaMapPtr(new TFColorRGBaMap(domain));
	clear();
}

TFRGBFunction::TFRGBFunction(TFRGBFunction &function){

	operator=(function);
}

TFRGBFunction::~TFRGBFunction(){
}

void TFRGBFunction::operator=(TFRGBFunction &function){

	type_ = function.getType();
	
	red_->clear();
	const TFColorRGBaMapPtr red = function.getRedFunction();
	TFColorRGBaMap::const_iterator beginRed = red->begin();
	TFColorRGBaMap::const_iterator endRed = red->end();
	for(TFColorRGBaMap::const_iterator it = beginRed; it!=endRed; ++it)
	{
		red_->push_back(*it);
	}

	green_->clear();
	const TFColorRGBaMapPtr green = function.getGreenFunction();
	TFColorRGBaMap::const_iterator beginGreen = green->begin();
	TFColorRGBaMap::const_iterator endGreen = green->end();
	for(TFColorRGBaMap::const_iterator it = beginGreen; it!=endGreen; ++it)
	{
		green_->push_back(*it);
	}

	blue_->clear();
	const TFColorRGBaMapPtr blue = function.getBlueFunction();
	TFColorRGBaMap::const_iterator beginBlue = blue->begin();
	TFColorRGBaMap::const_iterator endBlue = blue->end();
	for(TFColorRGBaMap::const_iterator it = beginBlue; it!=endBlue; ++it)
	{
		blue_->push_back(*it);
	}
}

TFAbstractFunction* TFRGBFunction::clone(){

	return new TFRGBFunction(*this);
}

void TFRGBFunction::clear(){

	TFColorRGBaMapIt beginRed = red_->begin();
	TFColorRGBaMapIt endRed = red_->end();
	for(TFColorRGBaMapIt it = beginRed; it!=endRed; ++it)
	{
		*it = 0;
	}

	TFColorRGBaMapIt beginGreen = green_->begin();
	TFColorRGBaMapIt endGreen = green_->end();
	for(TFColorRGBaMapIt it = beginGreen; it!=endGreen; ++it)
	{
		*it = 0;
	}

	TFColorRGBaMapIt beginBlue = blue_->begin();
	TFColorRGBaMapIt endBlue = blue_->end();
	for(TFColorRGBaMapIt it = beginBlue; it!=endBlue; ++it)
	{
		*it = 0;
	}
}

TFColorRGBaMapPtr TFRGBFunction::getRedFunction(){

	return red_;
}
TFColorRGBaMapPtr TFRGBFunction::getGreenFunction(){

	return green_;
}
TFColorRGBaMapPtr TFRGBFunction::getBlueFunction(){

	return blue_;
}

TFSize TFRGBFunction::getDomain(){

	return red_->size();
}

} // namespace GUI
} // namespace M4D
