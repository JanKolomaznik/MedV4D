#ifndef TF_COMMON
#define TF_COMMON

#include <sstream>
#include <exception>
#include <iostream>
#include <ostream>
#include <boost/shared_ptr.hpp>

#include <QtGui/QColor>

#include "MedV4D/GUI/TF/TFXmlWriterInterface.h"
#include "MedV4D/GUI/TF/TFXmlReaderInterface.h"

#include "MedV4D/Common/Common.h"

#define TF_DIMENSION_1 1

namespace M4D {
namespace GUI {
namespace TF{

typedef unsigned long Size;
typedef std::vector<int> Coordinates;

template<typename From, typename To>
inline To convert(const From &s){

	std::stringstream ss;
    To d;
    ss << s;
    if(ss >> d)
	{
        return d;
	}
    return To();
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
	bool operator!=(const Point& point){
		return !operator==(point);
	}
};
typedef Point<int, int> PaintingPoint;

struct AreaItem{
	TF::Size begin;
	TF::Size size;

	AreaItem():
		begin(0),
		size(0){
	}
	AreaItem(TF::Size begin, TF::Size size):
		begin(begin),
		size(size){
	}
};

template<typename ValueType>
void removeAllFromVector(typename std::vector<ValueType>& from, const ValueType& what){

	typename std::vector<ValueType> cleared;
	for(typename std::vector<ValueType>::iterator it = from.begin(); it != from.end(); ++it)
	{
		if(*it != what) cleared.push_back(*it);
	}
	std::swap(from, cleared);
}

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
