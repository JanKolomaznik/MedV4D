#include "TFGrayscaleTransparencyFunction.h"

namespace M4D {
namespace GUI {

TFGrayscaleTransparencyFunction::TFGrayscaleTransparencyFunction(TFSize domain){

	type_ = TFTYPE_GRAYSCALE_TRANSPARENCY;
	gray_ = TFColorRGBaMapPtr(new TFColorRGBaMap(domain));
	transparency_ = TFColorRGBaMapPtr(new TFColorRGBaMap(domain));
	clear();
}

TFGrayscaleTransparencyFunction::TFGrayscaleTransparencyFunction(TFGrayscaleTransparencyFunction &function){

	operator=(function);
}

TFGrayscaleTransparencyFunction::~TFGrayscaleTransparencyFunction(){
}

void TFGrayscaleTransparencyFunction::operator=(TFGrayscaleTransparencyFunction &function){

	type_ = function.getType();
	
	gray_->clear();
	const TFColorRGBaMapPtr gray = function.getGrayscaleFunction();
	TFColorRGBaMap::const_iterator beginGray = gray->begin();
	TFColorRGBaMap::const_iterator endGray = gray->end();
	for(TFColorRGBaMap::const_iterator it = beginGray; it!=endGray; ++it)
	{
		gray_->push_back(*it);
	}

	transparency_->clear();
	const TFColorRGBaMapPtr transparency = function.getTransparencyFunction();
	TFColorRGBaMap::const_iterator beginTr = transparency->begin();
	TFColorRGBaMap::const_iterator endTr = transparency->end();
	for(TFColorRGBaMap::const_iterator it = beginTr; it!=endTr; ++it)
	{
		transparency_->push_back(*it);
	}
}

TFAbstractFunction* TFGrayscaleTransparencyFunction::clone(){

	return new TFGrayscaleTransparencyFunction(*this);
}

void TFGrayscaleTransparencyFunction::clear(){

	TFColorRGBaMapIt beginGray = gray_->begin();
	TFColorRGBaMapIt endGray = gray_->end();
	for(TFColorRGBaMapIt it = beginGray; it!=endGray; ++it)
	{
		*it = 0;
	}

	TFColorRGBaMapIt beginTr = transparency_->begin();
	TFColorRGBaMapIt endTr = transparency_->end();
	for(TFColorRGBaMapIt it = beginTr; it!=endTr; ++it)
	{
		*it = 0;
	}
}

TFColorRGBaMapPtr TFGrayscaleTransparencyFunction::getGrayscaleFunction(){

	return gray_;
}
TFColorRGBaMapPtr TFGrayscaleTransparencyFunction::getTransparencyFunction(){

	return transparency_;
}

TFSize TFGrayscaleTransparencyFunction::getDomain(){

	return gray_->size();
}

} // namespace GUI
} // namespace M4D
