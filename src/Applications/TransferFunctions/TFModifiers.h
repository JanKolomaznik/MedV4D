#ifndef TF_MODIFIERS
#define TF_MODIFIERS

#include <TFPainters.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Modifier{
	ModifierSimple,
	ModifierPolygon,
	ModifierView
};
typedef std::vector<Modifier> Modifiers;

static Modifiers getAllowedModifiers(Painter painter){

	Modifiers allowed;

	switch(painter)
	{
		case PainterGrayscale:	//same as next case
		case PainterGrayscaleAlpha:	//same as next case
		case PainterRGB:	//same as next case
		case PainterRGBa:	//same as next case
		case PainterHSV:	//same as next case
		case PainterHSVa:
		{
			allowed.push_back(ModifierSimple);
			allowed.push_back(ModifierPolygon);
			allowed.push_back(ModifierView);
			break;
		}
	}

	return allowed;
}

}	//namespace Types


template<>
inline std::string convert<Types::Modifier, std::string>(const Types::Modifier &modifier){

	switch(modifier){
		case Types::ModifierSimple:
		{
			return "Simple modifier";
		}
		case Types::ModifierPolygon:
		{
			return "Polygon modifier";
		}
		case Types::ModifierView:
		{
			return "View modifier";
		}
	}

	tfAssert(!"Unknown modifier!");
	return "Unknown modifier (default)";
}

template<>
inline Types::Modifier TF::convert<std::string, Types::Modifier>(const std::string &modifier){

	if(modifier == "Simple modifier"){
		return Types::ModifierSimple;
	}
	if(modifier == "Polygon modifier"){
		return Types::ModifierPolygon;
	}
	if(modifier == "View modifier"){
		return Types::ModifierView;
	}

	tfAssert(!"Unknown modifier!");
	return Types::ModifierSimple;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_MODIFIERS