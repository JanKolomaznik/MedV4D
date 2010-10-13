#include "TFSimpleFunction.h"

TFSimpleFunction::TFSimpleFunction(unsigned functionRange, unsigned colorRange){

	functionRange_ = functionRange;
	colorRange_ = colorRange;
	type_ = TFTYPE_SIMPLE;
	name = "default_function";
	clear();
}

TFSimpleFunction::TFSimpleFunction(TFName functionName, unsigned functionRange, unsigned colorRange){

	functionRange_ = functionRange;
	colorRange_ = colorRange;
	type_ = TFTYPE_SIMPLE;
	name = functionName;
	clear();
}

TFSimpleFunction::TFSimpleFunction(TFSimpleFunction &function){

	operator=(function);
}

TFSimpleFunction::~TFSimpleFunction(){}

void TFSimpleFunction::operator=(TFSimpleFunction &function){

	type_ = function.getType();
	name = function.name;
	points_ = function.getPointMap();
}

TFAbstractFunction* TFSimpleFunction::clone(){

	return new TFSimpleFunction(*this);
}

void TFSimpleFunction::clear(){

	points_.clear();
	for(unsigned i = 0; i < functionRange_; ++i)
	{
		points_.push_back(0);
	}
}

void TFSimpleFunction::addPoint(int x, int y){

	points_[x] = y;
}

void TFSimpleFunction::addPoint(TFPoint point){

	points_[point.x] = point.y;
}

void TFSimpleFunction::addPoints(TFPoints points){

	TFPointsIterator first = points.begin();
	TFPointsIterator end = points.end();
	TFPointsIterator it = first;
	for(it; it != end; ++it)
	{
		addPoint(*it);
	}
}

void TFSimpleFunction::setPoints(TFPointMap points){

	points_ = points;
}

TFPoint TFSimpleFunction::getPoint(int coordX){

	return TFPoint(coordX, points_[coordX]);
}

TFPoints TFSimpleFunction::getAllPoints(){

	TFPoints points;
	for(unsigned i = 0; i < functionRange_; ++i)
	{
		points.push_back(TFPoint(i, points_[i]));
	}

	return points;
}

TFPointMap TFSimpleFunction::getPointMap(){

	return points_;
}

unsigned TFSimpleFunction::getFunctionRange(){

	return functionRange_;
}

unsigned TFSimpleFunction::getColorRange(){

	return colorRange_;
}

void TFSimpleFunction::recalculate(unsigned functionRange, unsigned colorRange){	//TODO

	TFPointMap newPoints(functionRange);
	apply(newPoints.begin(), functionRange, colorRange);

	functionRange_ = functionRange;
	colorRange_ = colorRange;
	points_ = newPoints;
}