#include "TFScheme.h"

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

void TFScheme::addFunction(TFName functionName, vector<TFPoint*> points){

	TFFunction* functionToAdd = new TFFunction(functionName);
	functionToAdd->addPointsFromSet(points);

	_functions->insert( make_pair(functionName, functionToAdd) );
}

void TFScheme::addFunction(TFFunction* function){

	_functions->insert( make_pair(function->name, function) );
}

void TFScheme::addFunctionsFromSet(vector<TFFunction*> functions){

	int functionCount = functions.size();
	for(int i = 0; i < functionCount; ++i)
	{
		addFunction(functions[i]);
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
		_functions->erase(functionName);
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
		return _functions->find(functionName)->second;
	}
	return NULL;
}