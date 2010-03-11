#ifndef TF_SCHEMEFUNCTION
#define TF_SCHEMEFUNCTION

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include <TF/Convert.h>
#include <TF/TFAFunction.h>

using namespace std;

struct TFSchemePoint{

	int x;
	int y;

	TFSchemePoint(): x(0), y(0){}
	TFSchemePoint(const TFSchemePoint &point): x(point.x), y(point.y){}
	TFSchemePoint(int x, int y): x(x), y(y){}
};

typedef map <int, TFSchemePoint*> TFSchemePoints;
typedef TFSchemePoints::iterator TFSchemePointsIterator;

class TFSchemeFunction{

public:
	TFName name;
	int colourRGB[3];

	TFSchemeFunction();
	TFSchemeFunction(TFName functionName);
	TFSchemeFunction(TFName functionName, int colour[3]);
	TFSchemeFunction(TFName functionName, int r, int g, int b);
	TFSchemeFunction(TFSchemeFunction &function);

	~TFSchemeFunction();

	void addPoint(int x, int y);
	void addPoint(TFSchemePoint* point, bool destroySource = true);

	void addPointsFromSet(vector<TFSchemePoint*> points, bool destroySource = true);

	bool containsPoint(int coordX);

	bool removePoint(int coordX);

	TFSchemePoint* getPoint(int coordX);

	//TFSchemePointsIterator begin();

	//TFSchemePointsIterator end();

	vector<TFSchemePoint *> getAllPoints();
	
private:	
	TFSchemePoints* _points;

};

#endif //TF_SCHEMEFUNCTION