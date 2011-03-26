#ifndef TF_FUNCTIONS
#define TF_FUNCTIONS

#include <TFHolders.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Function{
	FunctionRGB,
	FunctionHSV
};
typedef std::vector<Function> Functions;

static Functions getAllowedFunctions(Holder holder){

	Functions allowed;

	switch(holder){
		case Types::HolderBasic:
		{
			allowed.push_back(TF::Types::FunctionRGB);
			allowed.push_back(TF::Types::FunctionHSV);
			break;
		}
	}

	return allowed;
}

}	//namespace Types


template<>
inline std::string convert<Types::Function, std::string>(const Types::Function &function){

	switch(function){
		case Types::FunctionRGB:
		{
			return "RGB function";
		}
		case Types::FunctionHSV:
		{
			return "HSV function";
		}
	}

	tfAssert(!"Unknown function!");
	return "Unknown function (default)";
}

template<>
inline Types::Function TF::convert<std::string, Types::Function>(const std::string &function){

	if(function == "RGB function"){
		return Types::FunctionRGB;
	}
	if(function == "HSV function"){
		return Types::FunctionHSV;
	}

	tfAssert(!"Unknown function!");
	return Types::FunctionRGB;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_FUNCTIONS