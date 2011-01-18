#ifndef TF_TYPES
#define TF_TYPES

#include <sstream>
#include <exception>
#include <iostream>
#include <ostream>
#include <boost/shared_ptr.hpp>

#include <QtGui/QColor>

#include "common/Common.h"

namespace M4D {
namespace GUI {

enum TFFunctionType{
	TFFUNCTION_UNKNOWN,
	TFFUNCTION_RGBA,
	TFFUNCTION_HSVA
};	//TODO remove

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
static std::string convert<TFFunctionType, std::string>(const TFFunctionType &tfType){

	switch(tfType){
		case TFFUNCTION_RGBA:
		{
			return "TFHolderRGBa";
		}
		case TFFUNCTION_HSVA:
		{
			return "TFHolderHSVa";
		}
	}
	return "Unknown";
}

template<>
static TFFunctionType convert<std::string, TFFunctionType>(const std::string &tfType){

	if(tfType == "TFHolderRGBa"){
		return TFFUNCTION_RGBA;
	}
	if(tfType == "TFHolderHSVa"){
		return TFFUNCTION_HSVA;
	}
	return TFFUNCTION_UNKNOWN;
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

struct TFColor{
	float component1, component2, component3, alpha;

	TFColor(): component1(0), component2(0), component3(0), alpha(0){}
	TFColor(const TFColor &color): component1(color.component1), component2(color.component2), component3(color.component3), alpha(color.alpha){}
	TFColor(float component1, float component2, float component3, float alpha): component1(component1), component2(component2), component3(component3), alpha(alpha){}

	bool operator==(const TFColor& color){
		return (component1 == color.component1) && (component2 == color.component2) && (component3 == color.component3) && (alpha == color.alpha);
	}
};

typedef std::vector<TFColor> TFColorMap;
typedef TFColorMap::iterator TFColorMapIt;
typedef boost::shared_ptr<TFColorMap> TFColorMapPtr;

struct TFArea{	
	TFSize x, y, width, height;

	TFArea():
		x(0), y(0), width(0), height(0){}
	TFArea(TFSize x, TFSize y, TFSize width, TFSize height):
		x(x), y(y), width(width), height(height){}
};

enum MouseButton{
	MouseButtonLeft,
	MouseButtonRight,
	MouseButtonMid
};

#ifndef ROUND
#define ROUND(a) ( (int)(a+0.5) )
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

// if TF_NDEBUG macro is defined, switch off debugging support
#ifndef TF_NDEBUG

#define tfAssert(e) ((void)((!!(e))||(TFAbort((#e),__FILE__,__LINE__),0)))
#define tfAbort(e) TFAbort((#e),__FILE__,__LINE__)

#else

#define tfAssert(ignore) ((void)0)

#endif

} // namespace GUI
} // namespace M4D

#endif //TF_TYPES