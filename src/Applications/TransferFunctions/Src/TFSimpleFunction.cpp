#include "TFSimpleFunction.h"

TFSimpleFunction::TFSimpleFunction(int functionRange, int colorRange){

	functionRange_ = functionRange;
	colorRange_ = colorRange;
	type_ = TFTYPE_SIMPLE;
	name = "default_function";
	clear();
}

TFSimpleFunction::TFSimpleFunction(TFName functionName, int functionRange, int colorRange){

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
	for(int i = 0; i < functionRange_; ++i)
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
	for(int i = 0; i < functionRange_; ++i)
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

void TFSimpleFunction::recalculate(int functionRange, int colorRange){

	TFPointMap newPoints(functionRange);
	adjustBySimpleFunction(this, &newPoints, 0, colorRange, functionRange);

	functionRange_ = functionRange;
	colorRange_ = colorRange;
	points_ = newPoints;

	/*
	while(points_.size() >= functionRange_)
	{
		points_.pop_back();
	}
	while(points_.size() <= functionRange_)
	{
		points_.push_back(0);
	}
	for(int i = 0; i <= functionRange_; ++i)
	{
		if(points_[i] > colorRange_) points_[i] = colorRange_;
	}
	*/
}