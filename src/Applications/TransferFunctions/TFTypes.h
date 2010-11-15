#ifndef TF_TYPES
#define TF_TYPES

#include <sstream>
#include <exception>
#include <iostream>
#include <ostream>

#include "common/Common.h"

namespace M4D {
namespace GUI {

enum TFFunctionType{
	TFFUNCTION_UNKNOWN,
	TFFUNCTION_RGBA,
	TFFUNCTION_HSVA
};

enum TFHolderType{
	TFHOLDER_UNKNOWN,
	TFHOLDER_GRAYSCALE,
	TFHOLDER_GRAYSCALE_ALPHA,
	TFHOLDER_RGB,
	TFHOLDER_RGBA,
	TFHOLDER_HSV,
	TFHOLDER_HSVA
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
static std::string convert<TFHolderType, std::string>(const TFHolderType &holderType){

	switch(holderType){
		case TFHOLDER_GRAYSCALE:
		{
			return "Grayscale";
		}
		case TFHOLDER_GRAYSCALE_ALPHA:
		{
			return "Grayscale-alpha";
		}
		case TFHOLDER_RGB:
		{
			return "RGB";
		}
		case TFHOLDER_RGBA:
		{
			return "RGBa";
		}
		case TFHOLDER_HSV:
		{
			return "HSV";
		}
		case TFHOLDER_HSVA:
		{
			return "HSVa";
		}
	}
	return "Unknown";
}

template<>
static TFHolderType convert<std::string, TFHolderType>(const std::string &holderType){

	if(holderType == "Grayscale"){
		return TFHOLDER_GRAYSCALE;
	}
	if(holderType == "Grayscale-alpha"){
		return TFHOLDER_GRAYSCALE_ALPHA;
	}
	if(holderType == "RGB"){
		return TFHOLDER_RGB;
	}
	if(holderType == "RGBa"){
		return TFHOLDER_RGBA;
	}
	if(holderType == "HSV"){
		return TFHOLDER_HSV;
	}
	if(holderType == "HSVa"){
		return TFHOLDER_HSVA;
	}
	return TFHOLDER_UNKNOWN;
}

template<>
static std::string convert<TFFunctionType, std::string>(const TFFunctionType &tfType){

	switch(tfType){
		case TFFUNCTION_RGBA:
		{
			return "RGBa";
		}
		case TFFUNCTION_HSVA:
		{
			return "HSVa";
		}
	}
	return "Unknown";
}

template<>
static TFFunctionType convert<std::string, TFFunctionType>(const std::string &tfType){

	if(tfType == "RGBa"){
		return TFFUNCTION_RGBA;
	}
	if(tfType == "HSVa"){
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