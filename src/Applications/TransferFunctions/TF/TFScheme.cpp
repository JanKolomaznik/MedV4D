#include "TFScheme.h"

TFScheme::TFScheme(){

	name = "Default Scheme";
	_functions = new TFFunctions();
	TFFunction* defaultFunction = new TFFunction();
	_functions->insert(make_pair(defaultFunction->name, defaultFunction));
}

TFScheme::TFScheme(TFName schemeName): name(schemeName){

	_functions = new TFFunctions();
}
TFScheme::~TFScheme(){

	TFFunctionsIterator first = _functions->begin();
	TFFunctionsIterator end = _functions->end();

	for(TFFunctionsIterator it = first; it != end; ++it)
	{
		delete it->second;
	}

	delete _functions;
}

void TFScheme::addFunction(TFName functionName, vector<TFPoint*> points, bool destroySource){

	TFFunction* functionToAdd = new TFFunction(functionName);
	functionToAdd->addPointsFromSet(points, destroySource);

	_functions->insert( make_pair(functionName, functionToAdd) );
}

void TFScheme::addFunction(TFFunction* function, bool destroySource){

	_functions->insert( make_pair(function->name, new TFFunction(*function)) );
	if(destroySource)
	{
		delete function;
	}
}

void TFScheme::addFunctionsFromSet(vector<TFFunction*> functions, bool destroySource){

	vector<TFFunction*>::iterator first = functions.begin();
	vector<TFFunction*>::iterator end = functions.end();
	vector<TFFunction*>::iterator it = first;
	for(it; it != end; ++it)
	{
		addFunction(*it, destroySource);
	}
	if(destroySource)
	{
		functions.clear();
	}
}

bool TFScheme::containsFunction(TFName functionName){

	return _functions->find(functionName) != _functions->end();
}

bool TFScheme::removeFunction(TFName functionName){

	if(containsFunction(functionName))
	{
		TFFunctionsIterator toRemove = _functions->find(functionName);
		delete toRemove->second;
		_functions->erase(toRemove);
		return true;
	}
	return false;
}

void TFScheme::changeFunctionName(TFName from, TFName to){
	TFFunction* temp = _functions->find(from)->second;
	_functions->erase(temp->name);
	temp->name = to;
	_functions->insert( make_pair(to, temp) );
}

TFFunction* TFScheme::getFunction(TFName functionName){

	if(containsFunction(functionName))
	{
		return new TFFunction( *(_functions->find(functionName)->second) );
	}
	return NULL;
}

TFFunction* TFScheme::getFirstFunction(){

	return new TFFunction( *(_functions->begin()->second) );
}

vector<TFName> TFScheme::getFunctionNames(){

	vector<TFName> names;

	TFFunctionsIterator first = _functions->begin();
	TFFunctionsIterator end = _functions->end();
	for(TFFunctionsIterator it = first; it != end; ++it)
	{
		names.push_back(it->second->name);
	}

	return names;
}

int TFScheme::getValue(int x){
	return 0;	//TODO
}
/*
void TFScheme::paintEvent(QPaintEvent *e){
	_painter->paintEvent(e);
}

void TFScheme::mouseMoveEvent(QMouseEvent *e){
	_painter->mouseMoveEvent(e);
}

void TFScheme::mousePressEvent(QMouseEvent *e){
	_painter->mousePressEvent(e);
}
*/