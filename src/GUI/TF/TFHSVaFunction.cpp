#include "MedV4D/GUI/TF/TFHSVaFunction.h"

namespace M4D {
namespace GUI {

/*
template<TF::Size dim>
TFHSVaFunction<dim>::TFHSVaFunction(const TF::Size domain){

	colorMap_ = TF::MultiDColor<dim>::Map::Ptr(new TF::MultiDColor<dim>::Map(domain));
	domain_ = domain;
	clear();
}

template<TF::Size dim>
TFHSVaFunction<dim>::TFHSVaFunction(TFHSVaFunction<dim> &function){

	operator=(function);
}

template<TF::Size dim>
TFHSVaFunction<dim>::~TFHSVaFunction(){}

template<TF::Size dim>
TF::Color TFHSVaFunction<dim>::getRGBfColor(const TF::Size value, const TF::Size dimension){

	TF::Color rgbColor();

	QColor color;
	color.setHsvF(
		(*colorMap_)[value][dimension].component1,
		(*colorMap_)[value][dimension].component2,
		(*colorMap_)[value][dimension].component3,
		(*colorMap_)[value][dimension].alpha);

	rgbColor = TF::Color(color.redF(), color.greenF(), color.blueF(), color.alphaF());
	
	return rgbColor;
}

template<TF::Size dim>
typename TFAbstractFunction<dim>::Ptr TFHSVaFunction<dim>::clone(){

	return TFAbstractFunction<dim>::Ptr(new TFHSVaFunction<dim>(*this));
}
*/
} // namespace GUI
} // namespace M4D
