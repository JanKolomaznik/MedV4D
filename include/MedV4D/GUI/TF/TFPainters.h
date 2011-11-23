#ifndef TF_PAINTERS
#define TF_PAINTERS

#include "GUI/TF/TFFunctions.h"

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Painter{
	PainterRGBa1D,
	PainterHSVa1D,
	PainterGrayscaleAlpha1D
};
typedef std::vector<Painter> Painters;

inline Painters getAllowedPainters(Function function){

	Painters allowed;

	switch(function)
	{
		case TF::Types::FunctionRGBa:
		{
			allowed.push_back(TF::Types::PainterGrayscaleAlpha1D);
			allowed.push_back(TF::Types::PainterRGBa1D);
			break;
		}
		case TF::Types::FunctionHSVa:
		{
			allowed.push_back(TF::Types::PainterHSVa1D);
			break;
		}
	}

	return allowed;
}

}	//namespace Types


template<>
inline std::string convert<Types::Painter, std::string>(const Types::Painter &painter){

	switch(painter){
		case Types::PainterGrayscaleAlpha1D:
		{
			return "Grayscale-alpha 1D painter";
		}
		case Types::PainterRGBa1D:
		{
			return "RGBa 1D painter";
		}
		case Types::PainterHSVa1D:
		{
			return "HSVa 1D painter";
		}
	}
	
	tfAssert(!"Unknown painter!");
	return "Unknown painter (default)";
}

template<>
inline Types::Painter convert<std::string, Types::Painter>(const std::string &painter){

	if(painter == "Grayscale-alpha 1D painter"){
		return Types::PainterGrayscaleAlpha1D;
	}
	if(painter == "RGBa 1D painter"){
		return Types::PainterRGBa1D;
	}
	if(painter == "HSVa 1D painter"){
		return Types::PainterHSVa1D;
	}
	tfAssert(!"Unknown painter!");
	return Types::PainterRGBa1D;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_PAINTERS
