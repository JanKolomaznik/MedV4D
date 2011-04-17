#ifndef TF_FUNCTIONS
#define TF_FUNCTIONS

#include <TFModifiers.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Function{
	FunctionRGBa1D,
	FunctionHSVa1D
};
typedef std::vector<Function> Functions;

static Functions getAllowedFunctions(Modifier modifier){

	Functions allowed;

	switch(modifier){
		case ModifierSimple1D:
		case ModifierPolygon1D:
		case ModifierComposite1D:
		{
			//1D
			allowed.push_back(FunctionRGBa1D);
			allowed.push_back(FunctionHSVa1D);
			break;
		}
	}

	return allowed;
}

}	//namespace Types


template<>
inline std::string convert<Types::Function, std::string>(const Types::Function &function){

	switch(function){
		case Types::FunctionRGBa1D:
		{
			return "RGBa 1D function";
		}
		case Types::FunctionHSVa1D:
		{
			return "HSVa 1D function";
		}
	}

	tfAssert(!"Unknown function!");
	return "Unknown function (default)";
}

template<>
inline Types::Function TF::convert<std::string, Types::Function>(const std::string &function){

	if(function == "RGB 1D function"){
		return Types::FunctionRGBa1D;
	}
	if(function == "HSVa 1D function"){
		return Types::FunctionHSVa1D;
	}

	tfAssert(!"Unknown function!");
	return Types::FunctionRGBa1D;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_FUNCTIONS