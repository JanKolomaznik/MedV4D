#include <TFAbstractFunction.h>


namespace M4D {
namespace GUI {

TFAbstractFunction::TFAbstractFunction(): type_(TFFUNCTION_UNKNOWN){}

TFAbstractFunction::~TFAbstractFunction(){}

TFFunctionType TFAbstractFunction::getType() const{

	return type_;
}

TFSize TFAbstractFunction::getDomain(){

	return colorMap_->size();
}

TFColorMapPtr TFAbstractFunction::getColorMap(){

	return colorMap_;
}

void TFAbstractFunction::clear(){

	TFColorMapIt begin = colorMap_->begin();
	TFColorMapIt end = colorMap_->end();
	for(TFColorMapIt it = begin; it!=end; ++it)
	{
		it->component1 = 0;
		it->component2 = 0;
		it->component3 = 0;
		it->alpha = 0;
	}
}

void TFAbstractFunction::operator=(TFAbstractFunction &function){

	type_ = function.getType();
	
	colorMap_->clear();

	const TFColorMapPtr colorMap = function.getColorMap();

	TFColorMap::const_iterator begin = colorMap->begin();
	TFColorMap::const_iterator end = colorMap->end();
	for(TFColorMap::const_iterator it = begin; it!=end; ++it)
	{
		colorMap_->push_back(*it);
	}
}

} // namespace GUI
} // namespace M4D