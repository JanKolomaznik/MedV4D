#ifndef TF_PREDEFINED
#define TF_PREDEFINED

#include <TFHolders.h>
#include <TFFunctions.h>
#include <TFPainters.h>
#include <TFModifiers.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Predefined{
	PredefinedCustom,
	PredefinedLoad,
	PredefinedSimpleGrayscaleAlpha,
	PredefinedSimpleRGBa,
	PredefinedPolygonRGBa,
	PredefinedSimpleHSVa
};
typedef std::vector<Predefined> PredefinedTypes;

static PredefinedTypes getPredefinedTypes(){

	PredefinedTypes all;

	all.push_back(PredefinedLoad);
	all.push_back(PredefinedCustom);

	all.push_back(PredefinedSimpleGrayscaleAlpha);
	all.push_back(PredefinedSimpleRGBa);
	all.push_back(PredefinedSimpleHSVa);
	all.push_back(PredefinedPolygonRGBa);

	return all;
}

struct Structure{

	Predefined predefined;

	Holder holder;
	Function function;
	Painter painter;
	Modifier modifier;

	Structure():
		predefined(PredefinedCustom){
	}

	Structure(Predefined predefined, Holder holder, Function function, Painter painter, Modifier modifier):
		predefined(predefined),
		holder(holder),
		function(function),
		painter(painter),
		modifier(modifier){
	}
};

static Structure getPredefinedStructure(Predefined predefinedType){

	switch(predefinedType)
	{
		case PredefinedSimpleGrayscaleAlpha:
		{
			return Structure(PredefinedSimpleGrayscaleAlpha,
				HolderBasic,
				FunctionRGB,
				PainterGrayscaleAlpha,
				ModifierSimple);
		}
		case PredefinedSimpleRGBa:
		{
			return Structure(PredefinedSimpleRGBa,
				HolderBasic,
				FunctionRGB,
				PainterRGBa,
				ModifierSimple);
		}
		case PredefinedSimpleHSVa:
		{
			return Structure(PredefinedSimpleHSVa,
				HolderBasic,
				FunctionHSV,
				PainterHSVa,
				ModifierSimple);
		}
		case PredefinedPolygonRGBa:
		{
			return Structure(PredefinedPolygonRGBa,
				HolderBasic,
				FunctionRGB,
				PainterRGBa,
				ModifierPolygon);
		}
		case PredefinedCustom:
		{
			return Structure();
		}
	}

	tfAssert(!"Unknown predefined type");
	return Structure();	//custom
}

}	//namespace Types


template<>
inline std::string convert<Types::Predefined, std::string>(const Types::Predefined &predefined){

	switch(predefined){
		case Types::PredefinedCustom:
		{
			return "Custom";
		}
		case Types::PredefinedLoad:
		{
			return "Load";
		}
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
	}

	tfAssert(!"Unknown predefined type!");
	return "Custom";
}

template<>
inline Types::Predefined TF::convert<std::string, Types::Predefined>(const std::string &predefined){

	if(predefined == "Custom"){
		return Types::PredefinedCustom;
	}	
	if(predefined == "Load"){
		return Types::PredefinedLoad;
	}	
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