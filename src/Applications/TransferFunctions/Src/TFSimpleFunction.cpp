#include "TFSimpleFunction.h"

TFSimpleFunction::TFSimpleFunction(int functionRange, int colorRange){

	_functionRange = functionRange;
	_colorRange = colorRange;
	_type = TFTYPE_SIMPLE;
	name = "default_function";
	clear();
}

TFSimpleFunction::TFSimpleFunction(TFName functionName, int functionRange, int colorRange){

	_functionRange = functionRange;
	_colorRange = colorRange;
	_type = TFTYPE_SIMPLE;
	name = functionName;
	clear();
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
	for(int i = 0; i < _functionRange + 1; ++i)
	{
		_points.push_back(0);
	}
}

void TFSimpleFunction::addPoint(int x, int y){

	_points[x] = y;
}

void TFSimpleFunction::addPoint(TFPoint point){

	_points[point.x] = point.y;
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
/*
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
*/
TFPoint TFSimpleFunction::getPoint(int coordX){

	return TFPoint(coordX, _points[coordX]);
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

	TFPoints points;
	for(int i = 0; i < _functionRange; ++i)
	{
		points.push_back(TFPoint(i, _points[i]));
	}

	return points;
}

TFPointMap TFSimpleFunction::getPointMap(){

	return _points;
}

int TFSimpleFunction::getFunctionRange(){

	return _functionRange;
}

int TFSimpleFunction::getColorRange(){

	return _colorRange;
}

void TFSimpleFunction::recalculate(int functionRange, int colorRange){	

	_functionRange = functionRange;
	_colorRange = colorRange;

	while(_points.size() >= _functionRange)
	{
		_points.pop_back();
	}
	while(_points.size() <= _functionRange)
	{
		_points.push_back(0);
	}
	for(int i = 0; i <= _functionRange; ++i)
	{
		if(_points[i] > _colorRange) _points[i] = _colorRange;
	}
}