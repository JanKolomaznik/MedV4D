#ifndef TF_HSVA_FUNCTION
#define TF_HSVA_FUNCTION

#include <TFAbstractFunction.h>
#include <GUI/utils/TransferFunctionBuffer.h>
#include <QtGui/QColor>

namespace M4D {
namespace GUI {

template<TF::Size dim>
class TFHSVaFunction: public TFAbstractFunction<dim>{

public:

	TFHSVaFunction(const TF::Size domain){

		colorMap_ = TF::MultiDColor<dim>::Map::Ptr(new TF::MultiDColor<dim>::Map(domain));
		domain_ = domain;
		clear();
	}

	TFHSVaFunction(TFHSVaFunction<dim> &function){

		operator=(function);
	}

	~TFHSVaFunction(){}

	TF::Color getMappedRGBfColor(const TF::Size value, const TF::Size dimension){

		TF::Color rgbColor;

		QColor color;
		color.setHsvF(
			(*colorMap_)[value][dimension].component1,
			(*colorMap_)[value][dimension].component2,
			(*colorMap_)[value][dimension].component3,
			(*colorMap_)[value][dimension].alpha);

		rgbColor = TF::Color(color.redF(), color.greenF(), color.blueF(), color.alphaF());
		
		return rgbColor;
	}

	typename TFAbstractFunction<dim>::Ptr clone(){

		return TFAbstractFunction<dim>::Ptr(new TFHSVaFunction<dim>(*this));
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_FUNCTION