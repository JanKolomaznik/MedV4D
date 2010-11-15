#include "TFGrayscaleFunction.h"

namespace M4D {
namespace GUI {

TFGrayscaleFunction::TFGrayscaleFunction(TFSize domain){

	type_ = TFTYPE_GRAYSCALE;
	points_ = TFColorRGBaMapPtr(new TFColorRGBaMap(domain));
	clear();
}

TFGrayscaleFunction::TFGrayscaleFunction(TFGrayscaleFunction &function){

	operator=(function);
}

TFGrayscaleFunction::~TFGrayscaleFunction(){
}

void TFGrayscaleFunction::operator=(TFGrayscaleFunction &function){

	type_ = function.getType();
	const TFColorRGBaMapPtr points = function.getFunction();

	points_->clear();
	TFColorRGBaMap::const_iterator begin = points->begin();
	TFColorRGBaMap::const_iterator end = points->end();
	for(TFColorRGBaMap::const_iterator it = begin; it!=end; ++it)
	{
		points_->push_back(*it);
	}
}

TFAbstractFunction* TFGrayscaleFunction::clone(){

	return new TFGrayscaleFunction(*this);
}

void TFGrayscaleFunction::clear(){

	TFColorRGBaMapIt begin = points_->begin();
	TFColorRGBaMapIt end = points_->end();
	for(TFColorRGBaMapIt it = begin; it!=end; ++it)
	{
		*it = 0;
	}
}

void TFGrayscaleFunction::setPoint(TFSize point, float value){

	if(point > points_->size()) tfAbort("point out of range");
	(*points_)[point] = value;
}

void TFGrayscaleFunction::setFunction(TFColorRGBaMapPtr function){

	tfAssert(function->size() == points_->size());
	points_ = function;
}
/*
void TFGrayscaleFunction::setPoints(TFColorRGBaMap points){

	points_ = points;
}
float TFGrayscaleFunction::getPoint(TFSize point){

	return TFPointSimple(point, points_[point]);
}


TFPointsSimple TFGrayscaleFunction::getAllPoints(){

	TFPointsSimple points;

	for(TFSize i = 0; i < points_.size(); ++i)
	{
		points.push_back(TFPointSimple(i, points_[i]));
	}

	return points;
}
*/
TFColorRGBaMapPtr TFGrayscaleFunction::getFunction(){

	return points_;
}

TFSize TFGrayscaleFunction::getDomain(){

	return points_->size();
}
/*
void TFGrayscaleFunction::recalculate(unsigned functionRange, unsigned colorRange){	//TODO

	TFColorRGBaMap newPoints(functionRange);
	apply<TFPointMapIterator>(newPoints.begin(), newPoints.end(), colorRange);

	functionRange_ = functionRange;
	colorRange_ = colorRange;
	points_ = newPoints;
}
*/
} // namespace GUI
} // namespace M4D
