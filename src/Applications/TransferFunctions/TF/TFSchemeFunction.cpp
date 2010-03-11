#include "TFSchemeFunction.h"

using namespace std;

TFSchemeFunction::TFSchemeFunction(){

	name = "default_function";

	colourRGB[0] = 0;
	colourRGB[1] = 0;
	colourRGB[2] = 0;

	_points = new TFSchemePoints();
}

TFSchemeFunction::TFSchemeFunction(TFName functionName):
	name(functionName){

	colourRGB[0] = 0;
	colourRGB[1] = 0;
	colourRGB[2] = 0;

	_points = new TFSchemePoints();
}

TFSchemeFunction::TFSchemeFunction(TFName functionName, int colour[3]):
	name(functionName){

	colourRGB[0] = colour[0];
	colourRGB[1] = colour[1];
	colourRGB[2] = colour[2];

	_points = new TFSchemePoints();
}

TFSchemeFunction::TFSchemeFunction(TFName functionName, int r, int g, int b):
	name(functionName){	

	colourRGB[0] = r;
	colourRGB[1] = g;
	colourRGB[2] = b;

	_points = new TFSchemePoints();
}

TFSchemeFunction::TFSchemeFunction(TFSchemeFunction &function){
	name = function.name;

	colourRGB[0] = function.colourRGB[0];
	colourRGB[1] = function.colourRGB[1];
	colourRGB[2] = function.colourRGB[2];

	_points = new TFSchemePoints();

	addPointsFromSet(function.getAllPoints());
}

TFSchemeFunction::~TFSchemeFunction(){

	TFSchemePointsIterator first = _points->begin();
	TFSchemePointsIterator end = _points->end();
	for(TFSchemePointsIterator it = first; it != end; ++it)
	{
		delete it->second;
	}

	delete _points;
}

void TFSchemeFunction::addPoint(int x, int y){

	addPoint(new TFSchemePoint(x,y));
}

void TFSchemeFunction::addPoint(TFSchemePoint* point, bool destroySource){

	if(point != NULL)
	{
		TFSchemePointsIterator added = _points->find(point->x);
		if(added == _points->end())
		{
			_points->insert(make_pair(point->x, new TFSchemePoint(*point)));
		}
		else
		{
			added->second->y = point->y;
		}
		if(destroySource)
		{
			delete point;
		}
	}
}

void TFSchemeFunction::addPointsFromSet(vector<TFSchemePoint*> points, bool destroySource){

	if(!points.empty())
	{
		vector<TFSchemePoint*>::iterator first = points.begin();
		vector<TFSchemePoint*>::iterator end = points.end();
		vector<TFSchemePoint*>::iterator it = first;
		for(it; it != end; ++it)
		{
			addPoint(*it, destroySource);
		}
		if(destroySource)
		{
			points.clear();
		}
	}
}

bool TFSchemeFunction::containsPoint(int coordX){

	return _points->find(coordX) != _points->end();
}

bool TFSchemeFunction::removePoint(int coordX){

	if(containsPoint(coordX))
	{
		TFSchemePointsIterator toRemove = _points->find(coordX);
		delete toRemove->second;
		_points->erase(toRemove);
		return true;
	}
	return false;
}

TFSchemePoint* TFSchemeFunction::getPoint(int coordX){

	if(containsPoint(coordX))
	{
		return new TFSchemePoint(*(_points->find(coordX)->second));
	}
	return NULL;
}
/*
TFSchemePointsIterator TFSchemeFunction::begin(){
	return _points->begin();
}

TFSchemePointsIterator TFSchemeFunction::end(){
	return _points->end();
}
*/
vector<TFSchemePoint*> TFSchemeFunction::getAllPoints(){

	vector<TFSchemePoint*> points;

	if(_points != NULL && !_points->empty())
	{
		TFSchemePointsIterator first = _points->begin();
		TFSchemePointsIterator end = _points->end();

		for(TFSchemePointsIterator it = first; it != end; ++it)
		{
			points.push_back(new TFSchemePoint(*(it->second)));
		}
	}

	return points;
}