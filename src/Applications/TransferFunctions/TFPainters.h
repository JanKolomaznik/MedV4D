#ifndef TF_PAINTERS
#define TF_PAINTERS

#include <TFGrayscaleAlphaPainter.h>
#include <TFRGBaPainter.h>
#include <TFHSVaPainter.h>

namespace M4D {
namespace GUI {

namespace TF {
namespace Types {

enum Painter{
	PainterRGB,
	PainterHSV,
	PainterGrayscale,
	PainterRGBa,
	PainterHSVa,
	PainterGrayscaleAlpha
};
typedef std::vector<Painter> Painters;
typedef boost::shared_ptr<Painters> PaintersPtr;

static PaintersPtr getAllowedPainters(Function function){

	Painters* allowed = new Painters();

	switch(function)
	{
		case TF::Types::FunctionRGB:
		{
			allowed->push_back(TF::Types::PainterGrayscale);
			allowed->push_back(TF::Types::PainterGrayscaleAlpha);
			allowed->push_back(TF::Types::PainterRGB);
			allowed->push_back(TF::Types::PainterRGBa);
			break;
		}
		case TF::Types::FunctionHSV:
		{
			allowed->push_back(TF::Types::PainterHSV);
			allowed->push_back(TF::Types::PainterHSVa);
			break;
		}
	}

	return PaintersPtr(allowed);
}

template<Size dim>
static typename TFAbstractPainter<dim>::Ptr createPainter(Painter painter){

	switch(painter)
	{
		case TF::Types::PainterGrayscale:
		{
			return typename TFAbstractPainter<dim>::Ptr(new TFGrayscaleAlphaPainter(false));
		}
		case TF::Types::PainterGrayscaleAlpha:
		{
			return typename TFAbstractPainter<dim>::Ptr(new TFGrayscaleAlphaPainter(true));
		}
		case TF::Types::PainterRGB:
		{
			return typename TFAbstractPainter<dim>::Ptr(new TFRGBaPainter(false));
		}
		case TF::Types::PainterRGBa:
		{
			return typename TFAbstractPainter<dim>::Ptr(new TFRGBaPainter(true));
		}
		case TF::Types::PainterHSV:
		{
			return typename TFAbstractPainter<dim>::Ptr(new TFHSVaPainter(false));
		}
		case TF::Types::PainterHSVa:
		{
			return typename TFAbstractPainter<dim>::Ptr(new TFHSVaPainter(true));
		}
	}
	
	tfAssert(!"Unknown painter!");
	return typename TFAbstractPainter<dim>::Ptr(new TFRGBaPainter(true));
}

}	//namespace Types


template<>
inline std::string convert<Types::Painter, std::string>(const Types::Painter &painter){

	switch(painter){
		case Types::PainterGrayscale:
		{
			return "Grayscale painter";
		}
		case Types::PainterRGB:
		{
			return "RGB painter";
		}
		case Types::PainterHSV:
		{
			return "HSV painter";
		}
		case Types::PainterGrayscaleAlpha:
		{
			return "Grayscale-alpha painter";
		}
		case Types::PainterRGBa:
		{
			return "RGBa painter";
		}
		case Types::PainterHSVa:
		{
			return "HSVa painter";
		}
	}
	
	tfAssert(!"Unknown painter!");
	return "Unknown painter (default)";
}

template<>
inline Types::Painter TF::convert<std::string, Types::Painter>(const std::string &painter){

	if(painter == "Grayscale painter"){
		return Types::PainterGrayscale;
	}
	if(painter == "HSV painter"){
		return Types::PainterRGB;
	}
	if(painter == "HSV painter"){
		return Types::PainterHSV;
	}
	if(painter == "Grayscale-alpha painter"){
		return Types::PainterGrayscaleAlpha;
	}
	if(painter == "RGBa painter"){
		return Types::PainterRGBa;
	}
	if(painter == "HSVa painter"){
		return Types::PainterHSVa;
	}
	tfAssert(!"Unknown painter!");
	return Types::PainterRGBa;	//default
}

}	//namespace TF

}	//namespace GUI
}	//namespace M4D

#endif	//TF_PAINTERS