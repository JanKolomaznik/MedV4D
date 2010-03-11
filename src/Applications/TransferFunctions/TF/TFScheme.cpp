#include "TFScheme.h"
#include "ui_TFSchemeTools.h"

//class TFSchemePainter;
//class TFSchemeTools;
#include <TF/TFSchemePainter.h>
#include <TF/TFSchemeTools.h>

#define ROUND( a ) ( (int)(a+0.5) )

TFScheme::TFScheme(){

	name = "Default Scheme";
	_functions = new TFSchemeFunctions();
	currentFunction = new TFSchemeFunction();
	_functions->insert(make_pair(currentFunction->name, new TFSchemeFunction()));
	
	TFSchemePainter* painter = new TFSchemePainter(10, 10);
	painter->setView(this);
	_painter = painter;

	TFSchemeTools* tools = new TFSchemeTools();
	tools->setScheme(this);
	_tools = tools;

	QObject::connect(_tools, SIGNAL(CurrentFunctionChanged()), _painter, SLOT(Repaint()));
}

TFScheme::TFScheme(TFName schemeName): name(schemeName){

	_functions = new TFSchemeFunctions();
	
	TFSchemePainter* painter = new TFSchemePainter(10, 10);
	painter->setView(this);
	_painter = painter;

	TFSchemeTools* tools = new TFSchemeTools();
	tools->setScheme(this);
	_tools = tools;

	QObject::connect(_tools, SIGNAL(CurrentFunctionChanged()), _painter, SLOT(Repaint()));
}
TFScheme::~TFScheme(){

	TFSchemeFunctionsIterator first = _functions->begin();
	TFSchemeFunctionsIterator end = _functions->end();

	for(TFSchemeFunctionsIterator it = first; it != end; ++it)
	{
		delete it->second;
	}

	delete _functions;
	delete currentFunction;

	delete _painter;
	delete _tools;
}

void TFScheme::addFunction(TFName functionName, vector<TFSchemePoint*> points, bool destroySource){

	TFSchemeFunction* functionToAdd = new TFSchemeFunction(functionName);
	functionToAdd->addPointsFromSet(points, destroySource);

	_functions->insert( make_pair(functionName, functionToAdd) );
}

void TFScheme::addFunction(TFSchemeFunction* function, bool destroySource){

	if(function != NULL)
	{
		_functions->insert( make_pair(function->name, new TFSchemeFunction(*function)) );
		if(destroySource)
		{
			delete function;
		}
	}
}

void TFScheme::addFunctionsFromSet(vector<TFSchemeFunction*> functions, bool destroySource){

	if(!functions.empty())
	{
		vector<TFSchemeFunction*>::iterator first = functions.begin();
		vector<TFSchemeFunction*>::iterator end = functions.end();
		vector<TFSchemeFunction*>::iterator it = first;
		for(it; it != end; ++it)
		{
			addFunction(*it, destroySource);
		}
		if(destroySource)
		{
			functions.clear();
		}
	}
}

bool TFScheme::containsFunction(TFName functionName){

	return _functions->find(functionName) != _functions->end();
}

bool TFScheme::removeFunction(TFName functionName){

	if(containsFunction(functionName))
	{
		TFSchemeFunctionsIterator toRemove = _functions->find(functionName);
		delete toRemove->second;
		_functions->erase(toRemove);
		return true;
	}
	return false;
}

void TFScheme::changeFunctionName(TFName from, TFName to){

	if(containsFunction(from))
	{
		TFSchemeFunction* temp = _functions->find(from)->second;
		_functions->erase(temp->name);
		temp->name = to;
		_functions->insert( make_pair(to, temp) );
	}
}

TFSchemeFunction* TFScheme::getFunction(TFName functionName){

	if(containsFunction(functionName))
	{
		return new TFSchemeFunction( *(_functions->find(functionName)->second) );
	}
	return NULL;
}

TFSchemeFunction* TFScheme::getFirstFunction(){

	if(!_functions->empty())
	{
		return new TFSchemeFunction( *(_functions->begin()->second) );
	}
	return NULL;
}

vector<TFName> TFScheme::getFunctionNames(){

	vector<TFName> names;
	if(!_functions->empty())
	{
		TFSchemeFunctionsIterator first = _functions->begin();
		TFSchemeFunctionsIterator end = _functions->end();
		for(TFSchemeFunctionsIterator it = first; it != end; ++it)
		{
			names.push_back(it->second->name);
		}
	}
	return names;
}

void TFScheme::save(){
	((TFSchemeTools*)_tools)->save();
}

void TFScheme::load(){
	
	currentFunction = getFirstFunction();
	((TFSchemePainter*)_painter)->setView(this);
	((TFSchemeTools*)_tools)->setScheme(this);	
	((TFSchemeTools*)_tools)->load();
}

void TFScheme::adjustByTransferFunction(
		int* pixel,
		int min,
		int max,
		uint32 &width,
		uint32 &height,
		int brightnessRate,
		int contrastRate){
	
	if ( ! pixel || currentFunction == NULL)
	{
		return;
	}

	map<int, int> computed;
	vector<TFSchemePoint*> points = currentFunction->getAllPoints();

	if(points.empty())
	{
		return;
	}

	vector<TFSchemePoint*>::iterator first = points.begin();
	vector<TFSchemePoint*>::iterator end = points.end();
	vector<TFSchemePoint*>::iterator it = first;

	unsigned int i, j;

	for ( i = 0; i < height/2; ++i )
	{
		for ( j = 0; j < width/2; ++j )
		{
			int pixelValue = (pixel[ i * width + j ]/contrastRate)-brightnessRate;

			map<int, int>::iterator stored = computed.find(pixelValue);
			if(stored != computed.end())
			{
				pixel[ i * width + j ] = stored->second;
				continue;
			}

			int range = max - min;

			it = first;
			TFSchemePoint* lesser = *(it++);
			for(it; it != end; ++it)
			{
				if((*it)->x/(double)FUNCTION_RANGE > pixelValue/(double)range)
				{
					break;
				}
				lesser = *it;
			}
			double lesserValue, greaterValue;
			double distance;
			if(it == end)
			{
				greaterValue = min;
			}
			else
			{
				greaterValue = (*it)->y;
			}
			
			lesserValue = (lesser->y/(double)COLOUR_RANGE)*range;
			distance = ((pixelValue/(double)range)*FUNCTION_RANGE - lesser->x)/FUNCTION_RANGE;
			int result = ROUND(((greaterValue - lesserValue)*distance) + lesserValue);

			computed.insert(make_pair(pixelValue, result));

			pixel[ i * width + j ] = (result+brightnessRate)*contrastRate;
		}
	}
			
	it = first;
	for(it; it != end; ++it)
	{
		delete *it;
	}
}