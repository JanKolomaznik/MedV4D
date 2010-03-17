#ifndef TF_SIMPLEFUNCTION
#define TF_SIMPLEFUNCTION

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include <TFAbstractFunction.h>


static const int FUNCTION_RANGE_SIMPLE = 500;
static const int COLOR_RANGE_SIMPLE = 256;


class TFSimpleFunction: public TFAbstractFunction{

public:
	TFSimpleFunction();
	TFSimpleFunction(TFName functionName);
	TFSimpleFunction(TFSimpleFunction &function);

	~TFSimpleFunction();

	virtual TFAbstractFunction* clone();

	void addPoint(int x, int y);
	void addPoint(TFPoint point);

	void addPoints(TFPoints points);

	void setPoints(TFPointMap points);

	void clear();

	bool containsPoint(int coordX);

	bool removePoint(int coordX);

	TFPoint getPoint(int coordX);

	//TFPointMapIterator begin();

	//TFPointMapIterator end();

	TFPoints getAllPoints();

	TFPointMap getPointMap();
	
private:	
	TFPointMap _points;

};

#endif //TF_SIMPLEFUNCTION