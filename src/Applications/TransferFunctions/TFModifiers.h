#ifndef TF_MODIFIERS
#define TF_MODIFIERS

#include <TFPainters.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Modifier{
	ModifierSimple,
	ModifierPolygon
};
typedef std::vector<Modifier> Modifiers;

static Modifiers getAllowedModifiers(Painter painter){

	Modifiers allowed;

	switch(painter)
	{
		case TF::Types::PainterGrayscale:	//same as next case
		case TF::Types::PainterGrayscaleAlpha:	//same as next case
		case TF::Types::PainterRGB:	//same as next case
		case TF::Types::PainterRGBa:	//same as next case
		case TF::Types::PainterHSV:	//same as next case
		case TF::Types::PainterHSVa:
		{
			allowed.push_back(TF::Types::ModifierSimple);
			allowed.push_back(TF::Types::ModifierPolygon);
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

	tfAssert(!"Unknown modifier!");
	return Types::ModifierSimple;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_MODIFIERS