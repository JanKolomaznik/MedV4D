#ifndef TF_RGBA_FUNCTION
#define TF_RGBA_FUNCTION

#include <TFAbstractFunction.h>
#include <GUI/utils/TransferFunctionBuffer.h>

namespace M4D {
namespace GUI {

template<TF::Size dim>
class TFRGBaFunction: public TFAbstractFunction<dim>{

public:

	TFRGBaFunction(const TF::Size domain){

		colorMap_ = TF::MultiDColor<dim>::Map::Ptr(new TF::MultiDColor<dim>::Map(domain));
		domain_ = domain;
		clear();
	}

	TFRGBaFunction(TFRGBaFunction<dim> &function){

		operator=(function);
	}

	~TFRGBaFunction(){}

	TF::Color getMappedRGBfColor(const TF::Size value, const TF::Size dimension){

		return (*colorMap_)[value][dimension];
	}

	typename TFAbstractFunction<dim>::Ptr clone(){

		return TFAbstractFunction<dim>::Ptr(new TFRGBaFunction<dim>(*this));
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_FUNCTION