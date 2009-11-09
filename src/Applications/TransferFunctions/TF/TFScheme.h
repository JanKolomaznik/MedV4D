#ifndef TF_SCHEME
#define TF_SCHEME

#include "TFFunction.h"

#include <map>

using namespace std;

typedef map<TFName, TFFunction*> TFFunctions;
typedef TFFunctions::iterator TFFunctionsIterator;

class TFScheme{

public:
	TFName name;

	TFScheme(TFName schemeName);
	~TFScheme();

	void addFunction(TFName functionName, vector<TFPoint*> points);
	void addFunction(TFFunction* function);

	void addFunctionsFromSet(vector<TFFunction*> functions);

	bool containsFunction(TFName functionName);

	bool removeFunction(TFName functionName);

	void changeFunctionName(TFName from, TFName to);

	TFFunction* getFunction(TFName functionName);

private:
	TFFunctions* _functions;
};


#endif //TF_SCHEME