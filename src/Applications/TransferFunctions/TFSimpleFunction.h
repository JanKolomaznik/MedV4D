#ifndef TF_SIMPLEFUNCTION
#define TF_SIMPLEFUNCTION

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include <TFAbstractFunction.h>


class TFSimpleFunction: public TFAbstractFunction{

public:

	TFSimpleFunction(int functionRange = 500, int colorRange = 280);
	TFSimpleFunction(TFName functionName, int functionRange = 500, int colorRange = 280);
	TFSimpleFunction(TFSimpleFunction &function);

	~TFSimpleFunction();

	TFAbstractFunction* clone();

	void addPoint(int x, int y);
	void addPoint(TFPoint point);

	void addPoints(TFPoints points);

	void setPoints(TFPointMap points);

	void clear();

	//bool containsPoint(int coordX);

	//bool removePoint(int coordX);

	TFPoint getPoint(int coordX);

	//TFPointMapIterator begin();

	//TFPointMapIterator end();

	TFPoints getAllPoints();

	TFPointMap getPointMap();

	int getFunctionRange();

	int getColorRange();

	void recalculate(int functionRange, int colorRange);
	
private:	
	TFPointMap _points;

	int _functionRange;
	int _colorRange;
};

#endif //TF_SIMPLEFUNCTION