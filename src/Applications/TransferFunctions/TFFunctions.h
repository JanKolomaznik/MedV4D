#ifndef TF_FUNCTIONS
#define TF_FUNCTIONS

#include <TFRGBaFunction.h>
#include <TFHSVaFunction.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Function{
	FunctionRGB,
	FunctionHSV
};
typedef std::vector<Function> Functions;
typedef boost::shared_ptr<Functions> FunctionsPtr;

static FunctionsPtr getAllFunctions(){

	Functions* all = new Functions();
	all->push_back(TF::Types::FunctionRGB);
	all->push_back(TF::Types::FunctionHSV);

	return FunctionsPtr(all);
}

static TFAbstractFunction::Ptr createFunction(Function type, TF::Size domain){

	switch(type)
	{
		case TF::Types::FunctionRGB:
		{
			return TFAbstractFunction::Ptr(new TFRGBaFunction(domain));
		}
		case TF::Types::FunctionHSV:
		{
			return TFAbstractFunction::Ptr(new TFHSVaFunction(domain));
		}
	}

	tfAssert(!"Unknown function");
	return TFAbstractFunction::Ptr(new TFRGBaFunction(domain));	//default
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