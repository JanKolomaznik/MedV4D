#include "TFGrayscaleTransparencyFunction.h"

namespace M4D {
namespace GUI {

TFGrayscaleTransparencyFunction::TFGrayscaleTransparencyFunction(TFSize domain){

	type_ = TFTYPE_GRAYSCALE_TRANSPARENCY;
	gray_ = TFFunctionMapPtr(new TFFunctionMap(domain));
	transparency_ = TFFunctionMapPtr(new TFFunctionMap(domain));
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
	const TFFunctionMapPtr gray = function.getGrayscaleFunction();
	TFFunctionMap::const_iterator beginGray = gray->begin();
	TFFunctionMap::const_iterator endGray = gray->end();
	for(TFFunctionMap::const_iterator it = beginGray; it!=endGray; ++it)
	{
		gray_->push_back(*it);
	}

	transparency_->clear();
	const TFFunctionMapPtr transparency = function.getTransparencyFunction();
	TFFunctionMap::const_iterator beginTr = transparency->begin();
	TFFunctionMap::const_iterator endTr = transparency->end();
	for(TFFunctionMap::const_iterator it = beginTr; it!=endTr; ++it)
	{
		transparency_->push_back(*it);
	}
}

TFAbstractFunction* TFGrayscaleTransparencyFunction::clone(){

	return new TFGrayscaleTransparencyFunction(*this);
}

void TFGrayscaleTransparencyFunction::clear(){

	TFFunctionMapIt beginGray = gray_->begin();
	TFFunctionMapIt endGray = gray_->end();
	for(TFFunctionMapIt it = beginGray; it!=endGray; ++it)
	{
		*it = 0;
	}

	TFFunctionMapIt beginTr = transparency_->begin();
	TFFunctionMapIt endTr = transparency_->end();
	for(TFFunctionMapIt it = beginTr; it!=endTr; ++it)
	{
		*it = 0;
	}
}

TFFunctionMapPtr TFGrayscaleTransparencyFunction::getGrayscaleFunction(){

	return gray_;
}
TFFunctionMapPtr TFGrayscaleTransparencyFunction::getTransparencyFunction(){

	return transparency_;
}

TFSize TFGrayscaleTransparencyFunction::getDomain(){

	return gray_->size();
}

} // namespace GUI
} // namespace M4D
