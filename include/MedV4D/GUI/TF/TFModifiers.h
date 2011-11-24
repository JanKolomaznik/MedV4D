#ifndef TF_MODIFIERS
#define TF_MODIFIERS

#include "MedV4D/GUI/TF/TFDimensions.h"

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Modifier{
	ModifierSimple1D,
	ModifierPolygon1D,
	ModifierComposite1D
};
typedef std::vector<Modifier> Modifiers;

inline Modifiers getAllowedModifiers(Dimension dimension){

	Modifiers allowed;

	switch(dimension)
	{
		case Dimension1:
		{
			allowed.push_back(ModifierSimple1D);
			allowed.push_back(ModifierPolygon1D);
			allowed.push_back(ModifierComposite1D);
			break;
		}
	}

	return allowed;
}

}	//namespace Types


template<>
inline std::string convert<Types::Modifier, std::string>(const Types::Modifier &modifier){

	switch(modifier){
		case Types::ModifierSimple1D:
		{
			return "Free-hand 1D modifier";
		}
		case Types::ModifierPolygon1D:
		{
			return "Polygon 1D modifier";
		}
		case Types::ModifierComposite1D:
		{
			return "Composition 1D";
		}
	}

	tfAssert(!"Unknown modifier!");
	return "Unknown modifier (default)";
}

template<>
inline Types::Modifier convert<std::string, Types::Modifier>(const std::string &modifier){

	if(modifier == "Free-hand 1D modifier"){
		return Types::ModifierSimple1D;
	}
	if(modifier == "Polygon 1D modifier"){
		return Types::ModifierPolygon1D;
	}
	if(modifier == "Composition 1D"){
		return Types::ModifierComposite1D;
	}

	tfAssert(!"Unknown modifier!");
	return Types::ModifierSimple1D;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_MODIFIERS
