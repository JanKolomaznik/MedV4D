#include "MedV4D/GUI/TF/RGBaFunction.h"

namespace M4D {
namespace GUI {
/*
template<TF::Size dim>
RGBaFunction<dim>::RGBaFunction(const TF::Size domain){

	colorMap_ = TF::MultiDColor<dim>::Map::Ptr(new TF::MultiDColor<dim>::Map(domain));
	domain_ = domain;
	clear();
}

template<TF::Size dim>
RGBaFunction<dim>::RGBaFunction(RGBaFunction<dim> &function){

	operator=(function);
}

template<TF::Size dim>
RGBaFunction<dim>::~RGBaFunction(){}

template<TF::Size dim>
TF::Color RGBaFunction<dim>::getRGBfColor(const TF::Size value, const TF::Size dimension){

	return (*colorMap_)[value][dimension];
}

template<TF::Size dim>
typename AbstractFunction<dim>::Ptr RGBaFunction<dim>::clone(){

	return AbstractFunction<dim>::Ptr(new RGBaFunction<dim>(*this));
}
*/
} // namespace GUI
} // namespace M4D
