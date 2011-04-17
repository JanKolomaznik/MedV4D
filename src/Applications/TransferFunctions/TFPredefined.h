#ifndef TF_PREDEFINED
#define TF_PREDEFINED

#include <TFHolders.h>
#include <TFModifiers.h>
#include <TFFunctions.h>
#include <TFPainters.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Predefined{
	PredefinedCustom,
	PredefinedGrayscale1D,
	PredefinedRGBa1D,
	PredefinedHSVa1D,
	PredefinedPolygonRGBa1D,
	PredefinedComposition1D
};
typedef std::vector<Predefined> PredefinedTypes;

static PredefinedTypes getPredefinedTypes(){

	PredefinedTypes all;

	all.push_back(PredefinedGrayscale1D);
	all.push_back(PredefinedRGBa1D);
	all.push_back(PredefinedHSVa1D);
	all.push_back(PredefinedPolygonRGBa1D);
	all.push_back(PredefinedComposition1D);

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

	Structure(Predefined predefined, Holder holder, Modifier modifier, Function function, Painter painter):
		predefined(predefined),
		holder(holder),
		modifier(modifier),
		function(function),
		painter(painter){
	}
};

static Structure getPredefinedStructure(Predefined predefinedType){

	switch(predefinedType)
	{
		case PredefinedGrayscale1D:
		{
			return Structure(PredefinedGrayscale1D,
				HolderBasic,
				ModifierSimple1D,
				FunctionRGBa1D,
				PainterGrayscaleAlpha1D);
		}
		case PredefinedRGBa1D:
		{
			return Structure(PredefinedRGBa1D,
				HolderBasic,
				ModifierSimple1D,
				FunctionRGBa1D,
				PainterRGBa1D);
		}
		case PredefinedHSVa1D:
		{
			return Structure(PredefinedHSVa1D,
				HolderBasic,
				ModifierSimple1D,
				FunctionHSVa1D,
				PainterHSVa1D);
		}
		case PredefinedPolygonRGBa1D:
		{
			return Structure(PredefinedPolygonRGBa1D,
				HolderBasic,
				ModifierPolygon1D,
				FunctionRGBa1D,
				PainterRGBa1D);
		}
		case PredefinedComposition1D:
		{
			return Structure(PredefinedComposition1D,
				HolderBasic,
				ModifierComposite1D,
				FunctionRGBa1D,
				PainterRGBa1D);
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
		case Types::PredefinedGrayscale1D:
		{
			return "Grayscale-alpha 1D";
		}
		case Types::PredefinedRGBa1D:
		{
			return "RGBa 1D";
		}
		case Types::PredefinedHSVa1D:
		{
			return "HSVa 1D";
		}
		case Types::PredefinedPolygonRGBa1D:
		{
			return "Polygon RGBa 1D";
		}
		case Types::PredefinedComposition1D:
		{
			return "Composition 1D";
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
	if(predefined == "Grayscale-alpha 1D"){
		return Types::PredefinedGrayscale1D;
	}
	if(predefined == "RGBa 1D"){
		return Types::PredefinedRGBa1D;
	}
	if(predefined == "HSVa 1D"){
		return Types::PredefinedHSVa1D;
	}
	if(predefined == "Polygon RGBa 1D"){
		return Types::PredefinedPolygonRGBa1D;
	}	
	if(predefined == "Composition 1D"){
		return Types::PredefinedComposition1D;
	}	

	tfAssert(!"Unknown predefined type!");
	return Types::PredefinedCustom;
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_PREDEFINED