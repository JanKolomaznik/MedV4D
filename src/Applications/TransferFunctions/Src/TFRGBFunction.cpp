#include "TFRGBFunction.h"

namespace M4D {
namespace GUI {

TFRGBFunction::TFRGBFunction(TFSize domain){

	type_ = TFTYPE_RGB;
	red_ = TFFunctionMapPtr(new TFFunctionMap(domain));
	green_ = TFFunctionMapPtr(new TFFunctionMap(domain));
	blue_ = TFFunctionMapPtr(new TFFunctionMap(domain));
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
	const TFFunctionMapPtr red = function.getRedFunction();
	TFFunctionMap::const_iterator beginRed = red->begin();
	TFFunctionMap::const_iterator endRed = red->end();
	for(TFFunctionMap::const_iterator it = beginRed; it!=endRed; ++it)
	{
		red_->push_back(*it);
	}

	green_->clear();
	const TFFunctionMapPtr green = function.getGreenFunction();
	TFFunctionMap::const_iterator beginGreen = green->begin();
	TFFunctionMap::const_iterator endGreen = green->end();
	for(TFFunctionMap::const_iterator it = beginGreen; it!=endGreen; ++it)
	{
		green_->push_back(*it);
	}

	blue_->clear();
	const TFFunctionMapPtr blue = function.getBlueFunction();
	TFFunctionMap::const_iterator beginBlue = blue->begin();
	TFFunctionMap::const_iterator endBlue = blue->end();
	for(TFFunctionMap::const_iterator it = beginBlue; it!=endBlue; ++it)
	{
		blue_->push_back(*it);
	}
}

TFAbstractFunction* TFRGBFunction::clone(){

	return new TFRGBFunction(*this);
}

void TFRGBFunction::clear(){

	TFFunctionMapIt beginRed = red_->begin();
	TFFunctionMapIt endRed = red_->end();
	for(TFFunctionMapIt it = beginRed; it!=endRed; ++it)
	{
		*it = 0;
	}

	TFFunctionMapIt beginGreen = green_->begin();
	TFFunctionMapIt endGreen = green_->end();
	for(TFFunctionMapIt it = beginGreen; it!=endGreen; ++it)
	{
		*it = 0;
	}

	TFFunctionMapIt beginBlue = blue_->begin();
	TFFunctionMapIt endBlue = blue_->end();
	for(TFFunctionMapIt it = beginBlue; it!=endBlue; ++it)
	{
		*it = 0;
	}
}

TFFunctionMapPtr TFRGBFunction::getRedFunction(){

	return red_;
}
TFFunctionMapPtr TFRGBFunction::getGreenFunction(){

	return green_;
}
TFFunctionMapPtr TFRGBFunction::getBlueFunction(){

	return blue_;
}

TFSize TFRGBFunction::getDomain(){

	return red_->size();
}

} // namespace GUI
} // namespace M4D
