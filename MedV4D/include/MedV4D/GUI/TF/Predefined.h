#ifndef TF_PREDEFINED
#define TF_PREDEFINED

#include "MedV4D/GUI/TF/Dimensions.h"
#include "MedV4D/GUI/TF/Modifiers.h"
#include "MedV4D/GUI/TF/Functions.h"
#include "MedV4D/GUI/TF/Painters.h"

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

inline PredefinedTypes getPredefinedTypes(){

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

	Dimension dimension;
	Function function;
	Painter painter;
	Modifier modifier;

	Structure():
		predefined(PredefinedCustom){
	}

	Structure(Predefined predefined, Dimension dimension, Modifier modifier, Function function, Painter painter):
		predefined(predefined),
		dimension(dimension),
		function(function),
		painter(painter),
		modifier(modifier){
	}
};

inline Structure getPredefinedStructure(Predefined predefinedType){

	switch(predefinedType)
	{
		case PredefinedGrayscale1D:
		{
			return Structure(PredefinedGrayscale1D,
				Dimension1,
				ModifierSimple1D,
				FunctionRGBa,
				PainterGrayscaleAlpha1D);
		}
		case PredefinedRGBa1D:
		{
			return Structure(PredefinedRGBa1D,
				Dimension1,
				ModifierSimple1D,
				FunctionRGBa,
				PainterRGBa1D);
		}
		case PredefinedHSVa1D:
		{
			return Structure(PredefinedHSVa1D,
				Dimension1,
				ModifierSimple1D,
				FunctionHSVa,
				PainterHSVa1D);
		}
		case PredefinedPolygonRGBa1D:
		{
			return Structure(PredefinedPolygonRGBa1D,
				Dimension1,
				ModifierPolygon1D,
				FunctionRGBa,
				PainterRGBa1D);
		}
		case PredefinedComposition1D:
		{
			return Structure(PredefinedComposition1D,
				Dimension1,
				ModifierComposite1D,
				FunctionRGBa,
				PainterRGBa1D);
		}
		default:
			ASSERT( false );
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
inline Types::Predefined convert<std::string, Types::Predefined>(const std::string &predefined){

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
