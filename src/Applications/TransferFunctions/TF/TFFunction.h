#ifndef TF_FUNCTION
#define TF_FUNCTION

#include <map>
#include <vector>
#include <algorithm>
#include <string>

#include "Convert.h"

using namespace std;


typedef string TFName;

struct TFPoint{

	int x;
	int y;

	TFPoint(): x(0), y(0){}
	TFPoint(const TFPoint &point): x(point.x), y(point.y){}
	TFPoint(int x, int y): x(x), y(y){}

	static TFName makePointName(TFPoint* point){

		return "point("
			+ convert<int,string>(point->x) + "," + convert<int,string>(point->y)
			+ ")";
	}
};

typedef map <TFName, TFPoint*> TFPoints;
typedef TFPoints::iterator TFPointsIterator;

class TFFunction{

public:
	TFName name;
	int colourRGB[3];

	TFFunction(TFName functionName);
	TFFunction(TFName functionName, int colour[3]);
	TFFunction(TFName functionName, int r, int g, int b);

	~TFFunction();

	void addPoint(int x, int y);
	void addPoint(TFPoint* point);

	void addPointsFromSet(vector<TFPoint*> points);

	bool containsPoint(TFName pointName);

	bool removePoint(TFName pointName);

	TFName updatePoint(TFName pointName);

	TFPoint* getPoint(TFName pointName);

	vector<TFName> getPointNames();
	
private:	
	TFPoints* _points;

};

#endif //TF_FUNCTION