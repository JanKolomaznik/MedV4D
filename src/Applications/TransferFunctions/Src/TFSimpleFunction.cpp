#include "TFSimpleFunction.h"

TFSimpleFunction::TFSimpleFunction(){

	_type = TFTYPE_SIMPLE;
	name = "default_function";
}

TFSimpleFunction::TFSimpleFunction(TFName functionName){

	_type = TFTYPE_SIMPLE;
	name = functionName;
}

TFSimpleFunction::TFSimpleFunction(TFSimpleFunction &function){

	_type = function.getType();
	name = function.name;
	_points = function.getPointMap();
}

TFSimpleFunction::~TFSimpleFunction(){}

TFAbstractFunction* TFSimpleFunction::clone(){

	return new TFSimpleFunction(*this);
}

void TFSimpleFunction::clear(){

	_points.clear();
}

void TFSimpleFunction::addPoint(int x, int y){

	addPoint(TFPoint(x,y));
}

void TFSimpleFunction::addPoint(TFPoint point){

	TFPointMapIterator added = _points.find(point.x);
	if(added == _points.end())
	{
		_points.insert(std::make_pair(point.x, point));
	}
	else
	{
		added->second.y = point.y;
	}
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

	_points = points;
}

bool TFSimpleFunction::containsPoint(int coordX){

	return _points.find(coordX) != _points.end();
}

bool TFSimpleFunction::removePoint(int coordX){

	if(containsPoint(coordX))
	{
		TFPointMapIterator toRemove = _points.find(coordX);
		_points.erase(toRemove);
		return true;
	}
	return false;
}

TFPoint TFSimpleFunction::getPoint(int coordX){

	if(containsPoint(coordX))
	{
		return TFPoint(_points.find(coordX)->second);
	}
	return TFPoint(-1,-1);
}
/*
TFPointMapIterator TFSimpleFunction::begin(){
	return _points->begin();
}

TFPointMapIterator TFSimpleFunction::end(){
	return _points->end();
}
*/
TFPoints TFSimpleFunction::getAllPoints(){

	TFPointMapIterator first = _points.begin();
	TFPointMapIterator end = _points.end();

	TFPoints points;
	for(TFPointMapIterator it = first; it != end; ++it)
	{
		points.push_back(TFPoint(it->second));
	}

	return points;
}

TFPointMap TFSimpleFunction::getPointMap(){

	return _points;
}