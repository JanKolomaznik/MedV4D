#ifndef TF_PREDEFINED
#define TF_PREDEFINED

#include <TFFunctions.h>
#include <TFPainters.h>
#include <TFModifiers.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Predefined{
	PredefinedCustom,
	PredefinedSimpleGrayscaleAlpha,
	PredefinedSimpleRGBa,
	PredefinedPolygonRGBa,
	PredefinedSimpleHSVa
};
typedef std::vector<Predefined> PredefinedTypes;
typedef boost::shared_ptr<PredefinedTypes> PredefinedTypesPtr;

static PredefinedTypesPtr getPredefinedTypes(){

	PredefinedTypes* all = new PredefinedTypes();

	all->push_back(PredefinedCustom);

	all->push_back(PredefinedSimpleGrayscaleAlpha);
	all->push_back(PredefinedSimpleRGBa);
	all->push_back(PredefinedSimpleHSVa);
	all->push_back(PredefinedPolygonRGBa);

	return PredefinedTypesPtr(all);
}

struct PredefinedStructure{

	Predefined predefined;

	Function function;
	Painter painter;
	Modifier modifier;

	PredefinedStructure():
		predefined(PredefinedCustom){
	}

	PredefinedStructure(Predefined predefined, Function function, Painter painter, Modifier modifier):
		predefined(predefined),
		function(function),
		painter(painter),
		modifier(modifier){
	}
};

static PredefinedStructure getPredefinedStructure(Predefined predefinedType){

	switch(predefinedType)
	{
		case PredefinedSimpleGrayscaleAlpha:
		{
			return PredefinedStructure(PredefinedSimpleGrayscaleAlpha,
				FunctionRGB,
				PainterGrayscaleAlpha,
				ModifierSimple);
		}
		case PredefinedSimpleRGBa:
		{
			return PredefinedStructure(PredefinedSimpleRGBa,
				FunctionRGB,
				PainterRGBa,
				ModifierSimple);
		}
		case PredefinedSimpleHSVa:
		{
			return PredefinedStructure(PredefinedSimpleHSVa,
				FunctionHSV,
				PainterHSVa,
				ModifierSimple);
		}
		case PredefinedPolygonRGBa:
		{
			return PredefinedStructure(PredefinedPolygonRGBa,
				FunctionRGB,
				PainterRGBa,
				ModifierPolygon);
		}
		case PredefinedCustom:
		{
			return PredefinedStructure();
		}
	}

	tfAssert(!"Unknown predefined type");
	return PredefinedStructure();	//custom
}

}	//namespace Types


template<>
inline std::string convert<Types::Predefined, std::string>(const Types::Predefined &predefined){

	switch(predefined){
		case Types::PredefinedSimpleGrayscaleAlpha:
		{
			return "Grayscale-alpha";
		}
		case Types::PredefinedSimpleRGBa:
		{
			return "RGBa";
		}
		case Types::PredefinedSimpleHSVa:
		{
			return "HSVa";
		}
		case Types::PredefinedPolygonRGBa:
		{
			return "Polygon RGBa";
		}
		case Types::PredefinedCustom:
		{
			return "Custom";
		}
	}

	tfAssert(!"Unknown predefined type!");
	return "Custom";
}

template<>
inline Types::Predefined TF::convert<std::string, Types::Predefined>(const std::string &predefined){

	if(predefined == "Grayscale-alpha"){
		return Types::PredefinedSimpleGrayscaleAlpha;
	}
	if(predefined == "RGBa"){
		return Types::PredefinedSimpleRGBa;
	}
	if(predefined == "HSVa"){
		return Types::PredefinedSimpleHSVa;
	}
	if(predefined == "Polygon RGBa"){
		return Types::PredefinedPolygonRGBa;
	}	

	tfAssert(!"Unknown predefined type!");
	return Types::PredefinedCustom;
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_PREDEFINED