#ifndef TF_SCHEME
#define TF_SCHEME

#include <TF/TFFunction.h>

#include <map>

using namespace std;

typedef map<TFName, TFFunction*> TFFunctions;
typedef TFFunctions::iterator TFFunctionsIterator;

class TFScheme{

public:
	TFName name;

	TFScheme(TFName schemeName);
	~TFScheme();

	void addFunction(TFName functionName, vector<TFPoint*> points, bool destroySource = true);
	void addFunction(TFFunction* function, bool destroySource = true);

	void addFunctionsFromSet(vector<TFFunction*> functions, bool destroySource = true);

	bool containsFunction(TFName functionName);

	bool removeFunction(TFName functionName);

	void changeFunctionName(TFName from, TFName to);

	TFFunction* getFunction(TFName functionName);

private:
	TFFunctions* _functions;
};


#endif //TF_SCHEME