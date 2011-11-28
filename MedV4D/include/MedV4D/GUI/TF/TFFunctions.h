#ifndef TF_FUNCTIONS
#define TF_FUNCTIONS

#include "MedV4D/GUI/TF/TFModifiers.h"

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Function{
	FunctionRGBa,
	FunctionHSVa
};
typedef std::vector<Function> Functions;

inline Functions getAllowedFunctions(Modifier modifier){

	Functions allowed;

	switch(modifier){
		case ModifierSimple1D:
		case ModifierPolygon1D:
		case ModifierComposite1D:
		{
			//1D
			allowed.push_back(FunctionRGBa);
			allowed.push_back(FunctionHSVa);
			break;
		}
	}

	return allowed;
}

}	//namespace Types


template<>
inline std::string convert<Types::Function, std::string>(const Types::Function &function){

	switch(function){
		case Types::FunctionRGBa:
		{
			return "RGBa function";
		}
		case Types::FunctionHSVa:
		{
			return "HSVa function";
		}
	}

	tfAssert(!"Unknown function!");
	return "Unknown function (default)";
}

template<>
inline Types::Function convert<std::string, Types::Function>(const std::string &function){

	if(function == "RGBa function"){
		return Types::FunctionRGBa;
	}
	if(function == "HSVa function"){
		return Types::FunctionHSVa;
	}

	tfAssert(!"Unknown function!");
	return Types::FunctionRGBa;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_FUNCTIONS
