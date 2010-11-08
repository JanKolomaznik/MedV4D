#ifndef TF_TYPES
#define TF_TYPES

#include <sstream>
#include <exception>
#include <iostream>
#include <ostream>

#include <QtGui/QAction>
#include <QtGui/QMenu>
#include "common/Common.h"

namespace M4D {
namespace GUI {

enum TFType{
	TFTYPE_UNKNOWN,
	TFTYPE_SIMPLE,
	TFTYPE_GRAYSCALE_TRANSPARENCY,
	TFTYPE_RGB,
	TFTYPE_RGBA
};

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

template<>
static std::string convert<TFType, std::string>(const TFType &tfType){

	switch(tfType){
		case TFTYPE_SIMPLE:
		{
			return "Simple";
		}
		case TFTYPE_GRAYSCALE_TRANSPARENCY:
		{
			return "Grayscale-transparency";
		}
		case TFTYPE_RGB:
		{
			return "RGB";
		}
		case TFTYPE_RGBA:
		{
			return "RGBa";
		}
	}
	return "Unknown";
}

template<>
static TFType convert<std::string, TFType>(const std::string &tfType){

	if(tfType == "Simple"){
		return TFTYPE_SIMPLE;
	}
	if(tfType == "Grayscale-transparency"){
		return TFTYPE_GRAYSCALE_TRANSPARENCY;
	}
	if(tfType == "RGB"){
		return TFTYPE_RGB;
	}
	if(tfType == "RGBa"){
		return TFTYPE_RGBA;
	}
	return TFTYPE_UNKNOWN;
}

template <typename XType, typename YType>
struct TFPoint{

	XType x;
	YType y;

	TFPoint(): x(0), y(0){}
	TFPoint(const TFPoint<XType, YType> &point): x(point.x), y(point.y){}
	TFPoint(XType x, YType y): x(x), y(y){}

	bool operator==(const TFPoint& point){
		return (x == point.x) && (y == point.y);
	}
};

typedef TFPoint<int, int> TFPaintingPoint;

typedef std::string TFName;
typedef unsigned long TFSize;

typedef std::vector<float> TFFunctionMap;
typedef TFFunctionMap::iterator TFFunctionMapIt;
typedef boost::shared_ptr<TFFunctionMap> TFFunctionMapPtr;
/*
typedef std::vector<int> TFFunctionMap;
typedef TFFunctionMap::iterator TFFunctionMapIt;
*/
#ifndef ROUND
#define ROUND(a) ( (int)(a+0.5) )
#endif

// if TF_NDEBUG macro is defined, switch off debugging support
#ifndef TF_NDEBUG

#define tfAssert(e) ((void)((!!(e))||(TFAbort((#e),__FILE__,__LINE__),0)))
#define tfAbort(e) TFAbort((#e),__FILE__,__LINE__)

#else

#define tfAssert(ignore) ((void)0)

#endif

class TFAbortException : public std::exception
{
private:
    virtual const char * what() const throw()
    {
		return "TFAbortException";
    }
};

inline void TFAbort( const char * s, const char * f, int l)
{
	
    std::cout << f << "(" << l << "): " << s << std::endl;
    throw TFAbortException(); 
}

} // namespace GUI
} // namespace M4D

#endif //TF_TYPES