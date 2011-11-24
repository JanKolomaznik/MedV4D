#include "MedV4D/GUI/TF/TFRGBaFunction.h"

namespace M4D {
namespace GUI {
/*
template<TF::Size dim>
TFRGBaFunction<dim>::TFRGBaFunction(const TF::Size domain){

	colorMap_ = TF::MultiDColor<dim>::Map::Ptr(new TF::MultiDColor<dim>::Map(domain));
	domain_ = domain;
	clear();
}

template<TF::Size dim>
TFRGBaFunction<dim>::TFRGBaFunction(TFRGBaFunction<dim> &function){

	operator=(function);
}

template<TF::Size dim>
TFRGBaFunction<dim>::~TFRGBaFunction(){}

template<TF::Size dim>
TF::Color TFRGBaFunction<dim>::getRGBfColor(const TF::Size value, const TF::Size dimension){

	return (*colorMap_)[value][dimension];
}

template<TF::Size dim>
typename TFAbstractFunction<dim>::Ptr TFRGBaFunction<dim>::clone(){

	return TFAbstractFunction<dim>::Ptr(new TFRGBaFunction<dim>(*this));
}
*/
} // namespace GUI
} // namespace M4D
