#ifndef TF_FUNCTION
#define TF_FUNCTION

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>

#include <TF/Convert.h>
#include <TF/TFAFunction.h>

using namespace std;

struct TFPoint{

	int x;
	int y;

	TFPoint(): x(0), y(0){}
	TFPoint(const TFPoint &point): x(point.x), y(point.y){}
	TFPoint(int x, int y): x(x), y(y){}
};

typedef map <int, TFPoint*> TFPoints;
typedef TFPoints::iterator TFPointsIterator;

class TFFunction{

public:
	TFName name;
	int colourRGB[3];

	TFFunction();
	TFFunction(TFName functionName);
	TFFunction(TFName functionName, int colour[3]);
	TFFunction(TFName functionName, int r, int g, int b);
	TFFunction(TFFunction &function);

	~TFFunction();

	void addPoint(int x, int y);
	void addPoint(TFPoint* point, bool destroySource = true);

	void addPointsFromSet(vector<TFPoint*> points, bool destroySource = true);

	bool containsPoint(int coordX);

	bool removePoint(int coordX);

	TFPoint* getPoint(int coordX);

	//TFPointsIterator begin();

	//TFPointsIterator end();

	vector<TFPoint *> getAllPoints();

	void save(ofstream &outFile);
	
private:	
	TFPoints* _points;

};

#endif //TF_FUNCTION