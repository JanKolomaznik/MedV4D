#include "TFFunction.h"

using namespace std;

TFFunction::TFFunction(){

	name = "default_function";

	colourRGB[0] = 0;
	colourRGB[1] = 0;
	colourRGB[2] = 0;

	_points = new TFPoints();
}

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

TFFunction::TFFunction(TFFunction &function){
	name = function.name;

	colourRGB[0] = function.colourRGB[0];
	colourRGB[1] = function.colourRGB[1];
	colourRGB[2] = function.colourRGB[2];

	_points = new TFPoints();

	addPointsFromSet(function.getAllPoints());
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

void TFFunction::addPoint(TFPoint* point, bool destroySource){

	TFPointsIterator added = _points->find(point->x);
	if(added == _points->end())
	{
		_points->insert(make_pair(point->x, new TFPoint(*point)));
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

void TFFunction::addPointsFromSet(vector<TFPoint*> points, bool destroySource){

	vector<TFPoint*>::iterator first = points.begin();
	vector<TFPoint*>::iterator end = points.end();
	vector<TFPoint*>::iterator it = first;
	for(it; it != end; ++it)
	{
		addPoint(*it, destroySource);
	}
	if(destroySource)
	{
		points.clear();
	}
}

bool TFFunction::containsPoint(int coordX){

	return _points->find(coordX) != _points->end();
}

bool TFFunction::removePoint(int coordX){

	if(containsPoint(coordX))
	{
		TFPointsIterator toRemove = _points->find(coordX);
		delete toRemove->second;
		_points->erase(toRemove);
		return true;
	}
	return false;
}

TFPoint* TFFunction::getPoint(int coordX){

	if(containsPoint(coordX))
	{
		return new TFPoint(*(_points->find(coordX)->second));
	}
	return NULL;
}

TFPointsIterator TFFunction::begin(){
	return _points->begin();
}

TFPointsIterator TFFunction::end(){
	return _points->end();
}

vector<TFPoint*> TFFunction::getAllPoints(){

	TFPointsIterator first = _points->begin();
	TFPointsIterator end = _points->end();

	vector<TFPoint*> points;
	for(TFPointsIterator it = first; it != end; ++it)
	{
		points.push_back(new TFPoint(*(it->second)));
	}

	return points;
}

void TFFunction::save(ofstream &outFile){

	outFile << "\t<TFFunction name = \"" + name + "\" "
		<< "colourR = \"" + convert<int,string>(colourRGB[0]) + "\" "
		<< "colourG = \"" + convert<int,string>(colourRGB[1]) + "\" "
		<< "colourB = \"" + convert<int,string>(colourRGB[2]) + "\" >" << endl;

	TFPointsIterator first = _points->begin();
	TFPointsIterator end = _points->end();
	TFPointsIterator it = first;
	for(it; it != end; ++it)
	{
		outFile << "\t\t<TFPoint x = \"" + convert<int,string>(it->second->x) + "\" y = \"" + convert<int,string>(it->second->y) + "\" />" << endl;
	}

	outFile << "\t</TFFunction>" << endl;
}