#ifndef TF_COMMON
#define TF_COMMON

#include <sstream>
#include <exception>
#include <iostream>
#include <ostream>
#include <boost/shared_ptr.hpp>

#include <QtGui/QColor>

#include "common/Common.h"


namespace M4D {
namespace GUI {

namespace TF{

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

template <typename XType, typename YType>
struct Point{

	XType x;
	YType y;

	Point(): x(0), y(0){}
	Point(const Point<XType, YType> &point): x(point.x), y(point.y){}
	Point(XType x, YType y): x(x), y(y){}

	bool operator==(const Point& point){
		return (x == point.x) && (y == point.y);
	}
};
typedef Point<int, int> PaintingPoint;

typedef unsigned long Size;

//------------Debug------------------------------------------------------

class TFException : public std::exception
{
private:
    virtual const char * what() const throw()
    {
		return "TFException";
    }
};

inline void abort( const char * s, const char * f, int l)
{	
    std::cout << f << "(" << l << "): " << s << std::endl;
    throw TFException(); 
}

// if TF_NDEBUG macro is defined, switch off debugging support
#ifndef TF_NDEBUG

#define tfAssert(e) ((void)((!!(e))||(M4D::GUI::TF::abort((#e),__FILE__,__LINE__),0)))

#else

#define tfAssert(ignore) ((void)0)

#endif

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif //TF_COMMON
