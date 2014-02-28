#include "MedV4D/GUI/TF/HSVaFunction.h"

namespace M4D {
namespace GUI {

/*
template<TF::Size dim>
HSVaFunction<dim>::HSVaFunction(const TF::Size domain){

	colorMap_ = TF::MultiDColor<dim>::Map::Ptr(new TF::MultiDColor<dim>::Map(domain));
	domain_ = domain;
	clear();
}

template<TF::Size dim>
HSVaFunction<dim>::HSVaFunction(HSVaFunction<dim> &function){

	operator=(function);
}

template<TF::Size dim>
HSVaFunction<dim>::~HSVaFunction(){}

template<TF::Size dim>
TF::Color HSVaFunction<dim>::getRGBfColor(const TF::Size value, const TF::Size dimension){

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
typename AbstractFunction<dim>::Ptr HSVaFunction<dim>::clone(){

	return AbstractFunction<dim>::Ptr(new HSVaFunction<dim>(*this));
}
*/
} // namespace GUI
} // namespace M4D
