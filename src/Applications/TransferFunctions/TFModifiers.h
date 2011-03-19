#ifndef TF_MODIFIERS
#define TF_MODIFIERS

#include <TFSimpleModifier.h>
#include <TFPolygonModifier.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Modifier{
	ModifierSimple,
	ModifierPolygon
};
typedef std::vector<Modifier> Modifiers;
typedef boost::shared_ptr<Modifiers> ModifiersPtr;

static ModifiersPtr getAllowedModifiers(Painter painter){

	Modifiers* allowed = new Modifiers();

	switch(painter)
	{
		case TF::Types::PainterGrayscale:	//same as next case
		case TF::Types::PainterGrayscaleAlpha:	//same as next case
		case TF::Types::PainterRGB:	//same as next case
		case TF::Types::PainterRGBa:	//same as next case
		case TF::Types::PainterHSV:	//same as next case
		case TF::Types::PainterHSVa:
		{
			allowed->push_back(TF::Types::ModifierSimple);
			allowed->push_back(TF::Types::ModifierPolygon);
			break;
		}
	}

	return ModifiersPtr(allowed);
}

static TFAbstractModifier::Ptr createModifier(Modifier type, TFWorkCopy::Ptr workCopy, Painter painterUsed){

	switch(type)
	{
		case TF::Types::ModifierSimple:
		{
			switch(painterUsed)
			{
				case TF::Types::PainterGrayscale:
				{
					return TFAbstractModifier::Ptr(new TFSimpleModifier(workCopy, TFSimpleModifier::Grayscale, false));
				}
				case TF::Types::PainterGrayscaleAlpha:
				{
					return TFAbstractModifier::Ptr(new TFSimpleModifier(workCopy, TFSimpleModifier::Grayscale, true));
				}
				case TF::Types::PainterRGB:
				{
					return TFAbstractModifier::Ptr(new TFSimpleModifier(workCopy, TFSimpleModifier::RGB, false));
				}
				case TF::Types::PainterRGBa:
				{
					return TFAbstractModifier::Ptr(new TFSimpleModifier(workCopy, TFSimpleModifier::RGB, true));
				}
				case TF::Types::PainterHSV:
				{
					return TFAbstractModifier::Ptr(new TFSimpleModifier(workCopy, TFSimpleModifier::HSV, false));
				}
				case TF::Types::PainterHSVa:
				{
					return TFAbstractModifier::Ptr(new TFSimpleModifier(workCopy, TFSimpleModifier::HSV, true));
				}
			}
		}
		case TF::Types::ModifierPolygon:
		{
			switch(painterUsed)
			{
				case TF::Types::PainterGrayscale:
				{
					return TFAbstractModifier::Ptr(new TFPolygonModifier(workCopy, TFPolygonModifier::Grayscale, false));
				}
				case TF::Types::PainterGrayscaleAlpha:
				{
					return TFAbstractModifier::Ptr(new TFPolygonModifier(workCopy, TFPolygonModifier::Grayscale, true));
				}
				case TF::Types::PainterRGB:
				{
					return TFAbstractModifier::Ptr(new TFPolygonModifier(workCopy, TFPolygonModifier::RGB, false));
				}
				case TF::Types::PainterRGBa:
				{
					return TFAbstractModifier::Ptr(new TFPolygonModifier(workCopy, TFPolygonModifier::RGB, true));
				}
				case TF::Types::PainterHSV:
				{
					return TFAbstractModifier::Ptr(new TFPolygonModifier(workCopy, TFPolygonModifier::HSV, false));
				}
				case TF::Types::PainterHSVa:
				{
					return TFAbstractModifier::Ptr(new TFPolygonModifier(workCopy, TFPolygonModifier::HSV, true));
				}
			}
		}
	}

	tfAssert(!"Unknown modifier!");
	return TFAbstractModifier::Ptr(new TFSimpleModifier(workCopy, TFSimpleModifier::RGB, true));	//default
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