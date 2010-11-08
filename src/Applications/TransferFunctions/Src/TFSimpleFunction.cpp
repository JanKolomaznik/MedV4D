#include "TFSimpleFunction.h"

namespace M4D {
namespace GUI {

TFSimpleFunction::TFSimpleFunction(TFSize domain){

	type_ = TFTYPE_SIMPLE;
	points_ = TFFunctionMapPtr(new TFFunctionMap(domain));
	clear();
}

TFSimpleFunction::TFSimpleFunction(TFSimpleFunction &function){

	operator=(function);
}

TFSimpleFunction::~TFSimpleFunction(){
}

void TFSimpleFunction::operator=(TFSimpleFunction &function){

	type_ = function.getType();
	const TFFunctionMapPtr points = function.getFunction();

	points_->clear();
	TFFunctionMap::const_iterator begin = points->begin();
	TFFunctionMap::const_iterator end = points->end();
	for(TFFunctionMap::const_iterator it = begin; it!=end; ++it)
	{
		points_->push_back(*it);
	}
}

TFAbstractFunction* TFSimpleFunction::clone(){

	return new TFSimpleFunction(*this);
}

void TFSimpleFunction::clear(){

	TFFunctionMapIt begin = points_->begin();
	TFFunctionMapIt end = points_->end();
	for(TFFunctionMapIt it = begin; it!=end; ++it)
	{
		*it = 0;
	}
}

void TFSimpleFunction::setPoint(TFSize point, float value){

	if(point > points_->size()) tfAbort("point out of range");
	(*points_)[point] = value;
}

void TFSimpleFunction::setFunction(TFFunctionMapPtr function){

	tfAssert(function->size() == points_->size());
	points_ = function;
}
/*
void TFSimpleFunction::setPoints(TFFunctionMap points){

	points_ = points;
}
float TFSimpleFunction::getPoint(TFSize point){

	return TFPointSimple(point, points_[point]);
}


TFPointsSimple TFSimpleFunction::getAllPoints(){

	TFPointsSimple points;

	for(TFSize i = 0; i < points_.size(); ++i)
	{
		points.push_back(TFPointSimple(i, points_[i]));
	}

	return points;
}
*/
TFFunctionMapPtr TFSimpleFunction::getFunction(){

	return points_;
}

TFSize TFSimpleFunction::getDomain(){

	return points_->size();
}
/*
void TFSimpleFunction::recalculate(unsigned functionRange, unsigned colorRange){	//TODO

	TFFunctionMap newPoints(functionRange);
	apply<TFPointMapIterator>(newPoints.begin(), newPoints.end(), colorRange);

	functionRange_ = functionRange;
	colorRange_ = colorRange;
	points_ = newPoints;
}
*/
} // namespace GUI
} // namespace M4D
