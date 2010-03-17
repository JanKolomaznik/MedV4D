#ifndef TF_TYPES
#define TF_TYPES

#include <sstream>


typedef std::string TFName;

struct TFPoint{

	int x;
	int y;

	TFPoint(): x(0), y(0){}
	TFPoint(const TFPoint &point): x(point.x), y(point.y){}
	TFPoint(int x, int y): x(x), y(y){}
};

typedef std::vector<TFPoint> TFPoints;
typedef TFPoints::iterator TFPointsIterator;

typedef std::map <int, TFPoint> TFPointMap;
typedef TFPointMap::iterator TFPointMapIterator;


template<typename From, typename To>
static To convert(const From &s){

	std::stringstream ss;
    To d;
    ss << s;
    if(ss >> d)
	{
        return d;
	}
    return NULL;
}

enum TFType{
	TFTYPE_UNKNOWN,
	TFTYPE_SIMPLE
};

template<>
static std::string convert<TFType, std::string>(const TFType &tfType){

	switch(tfType){
		case TFTYPE_SIMPLE:
		{
			return "Simple";
		}
	}
	return "Unknown";
}

template<>
static TFType convert<std::string, TFType>(const std::string &tfType){

	if(tfType == "Simple"){
		return TFTYPE_SIMPLE;
	}
	return TFTYPE_UNKNOWN;
}
#endif //TF_TYPES