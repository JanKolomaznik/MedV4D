#ifndef TF_SCHEME
#define TF_SCHEME

#include <TF/TFAFunction.h>
#include <TF/TFSchemeFunction.h>

#include <map>
#include <fstream>	

using namespace std;

typedef map<TFName, TFSchemeFunction*> TFSchemeFunctions;
typedef TFSchemeFunctions::iterator TFSchemeFunctionsIterator;

class TFScheme: public TFAFunction{

public:
	TFName name;

	TFScheme();
	TFScheme(TFName schemeName);

	~TFScheme();

	void addFunction(TFName functionName, vector<TFSchemePoint*> points, bool destroySource = true);
	void addFunction(TFSchemeFunction* function, bool destroySource = true);
	void addFunctionsFromSet(vector<TFSchemeFunction*> functions, bool destroySource = true);

	bool containsFunction(TFName functionName);
	bool removeFunction(TFName functionName);
	void changeFunctionName(TFName from, TFName to);

	TFSchemeFunction* getFunction(TFName functionName);
	TFSchemeFunction* getFirstFunction();
	vector<TFName> getFunctionNames();

	virtual void adjustByTransferFunction(
		int* pixel,
		int min,
		int max,
		uint32 &width,
		uint32 &height,
		int brightnessRate,
		int contrastRate);

	virtual void save();

	virtual void load();

	TFSchemeFunction* currentFunction;

private:
	TFSchemeFunctions* _functions;
};


#endif //TF_SCHEME