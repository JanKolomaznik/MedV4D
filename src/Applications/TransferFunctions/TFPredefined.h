#ifndef TF_PREDEFINED
#define TF_PREDEFINED

#include <TFFunctions.h>
#include <TFPainters.h>
#include <TFModifiers.h>

#include <TFHolderInterface.h>
#include <TFHolder.h>

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

struct Structure{

	Predefined predefined;

	Size dimension;
	Function function;
	Painter painter;
	Modifier modifier;

	Structure():
		predefined(PredefinedCustom){
	}

	Structure(Predefined predefined, Size dimension, Function function, Painter painter, Modifier modifier):
		dimension(dimension),
		predefined(predefined),
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
			return Structure(
				PredefinedSimpleGrayscaleAlpha,
				1,
				FunctionRGB,
				PainterGrayscaleAlpha,
				ModifierSimple);
		}
		case PredefinedSimpleRGBa:
		{
			return Structure(PredefinedSimpleRGBa,
				1,
				FunctionRGB,
				PainterRGBa,
				ModifierSimple);
		}
		case PredefinedSimpleHSVa:
		{
			return Structure(PredefinedSimpleHSVa,
				1,
				FunctionHSV,
				PainterHSVa,
				ModifierSimple);
		}
		case PredefinedPolygonRGBa:
		{
			return Structure(PredefinedPolygonRGBa,
				1,
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

namespace Types{	

static TFHolderInterface* createHolder(QMainWindow* mainWindow, Size domain, Structure structure){
	
	switch(structure.function)
	{
		case FunctionRGB:	//TODO holder type
		case FunctionHSV:
		{
			TFAbstractFunction<1>::Ptr function = createFunction<1>(structure.function, domain);
			TFAbstractPainter<1>::Ptr painter = createPainter<1>(structure.painter);
			TFWorkCopy<1>::Ptr workCopy = TFWorkCopy<1>::Ptr(new TFWorkCopy<1>(function));
			TFAbstractModifier<1>::Ptr modifier = createModifier<1>(structure.modifier, workCopy, structure.painter);
			return new TFHolder(mainWindow,
				painter,
				modifier,
				convert<Predefined, std::string>(structure.predefined));
		}
	}

	tfAssert(!"Unknown holder");
	return NULL;
}

}	//namespace Types

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_PREDEFINED