#include "TFFunction.h"

using namespace std;

TFFunction::TFFunction(TFName functionName):
	name(functionName){

	colourRGB[0] = 0;
	colourRGB[1] = 0;
	colourRGB[2] = 0;
	_points = new TFPoints();
}

TFFunction::TFFunction(TFName functionName, int colour[3]):
	name(functionName){

	colourRGB[0] = colour[0];
	colourRGB[1] = colour[1];
	colourRGB[2] = colour[2];
	_points = new TFPoints();
}

TFFunction::TFFunction(TFName functionName, int r, int g, int b):
	name(functionName){	

	colourRGB[0] = r;
	colourRGB[1] = g;
	colourRGB[2] = b;
	_points = new TFPoints();
}

TFFunction::~TFFunction(){

	TFPointsIterator first = _points->begin();
	TFPointsIterator end = _points->end();
	for(TFPointsIterator it = first; it != end; ++it)
	{
		delete it->second;
	}

	delete _points;
}

void TFFunction::addPoint(int x, int y){

	addPoint(new TFPoint(x,y));
}

void TFFunction::addPoint(TFPoint* point){

	_points->insert(make_pair(TFPoint::makePointName(point), point));
}

void TFFunction::addPointsFromSet(vector<TFPoint*> points){

	int pointCount = points.size();
	for(int i = 0; i < pointCount; ++i)
	{
		addPoint(points[i]);
	}
}

bool TFFunction::containsPoint(TFName name){

	return _points->find(name) != _points->end();
}

bool TFFunction::removePoint(TFName name){

	if(containsPoint(name))
	{
		TFPointsIterator toRemove = _points->find(name);
		delete toRemove->second;
		_points->erase(name);
		return true;
	}
	return false;
}

TFName TFFunction::updatePoint(TFName pointName){
	TFPoint* temp = _points->find(pointName)->second;
	_points->erase(pointName);

	TFName newName = TFPoint::makePointName(temp);

	_points->insert( make_pair(newName, temp) );

	return newName;
}

TFPoint* TFFunction::getPoint(TFName name){

	if(containsPoint(name))
	{
		return _points->find(name)->second;
	}
	return NULL;
}

vector<TFName> TFFunction::getPointNames(){

	TFPointsIterator first = _points->begin();
	TFPointsIterator end = _points->end();

	vector<TFName> names;
	for(TFPointsIterator it = first; it != end; ++it)
	{
		names.push_back(it->first);
	}

	return names;
}