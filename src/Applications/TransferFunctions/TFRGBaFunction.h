#ifndef TF_RGBA_FUNCTION
#define TF_RGBA_FUNCTION

#include <TFAbstractFunction.h>

namespace M4D {
namespace GUI {
/*
template<TF::Size dim>
class TFRGBaFunction: public TFAbstractFunction<dim>{

public:

	TFRGBaFunction(const TF::Size domain[dim]){

		for(TF::Size i = 0; i < dim; ++i)
		{
			colorMap_[i] = TF::Color::MapPtr(new TF::Color::Map(domain[i]));
			clear(i);
		}
	}

	TFRGBaFunction(TFRGBaFunction<dim> &function){

		operator=(function);
	}

	~TFRGBaFunction(){}

	TF::Color getRGBfColor(const TF::Size index, const TF::Size dimension){

		return (*colorMap_[dimension])[index];
	}
	void setRGBfColor(const TF::Size index, const TF::Size dimension, const TF::Color& value){

		(*colorMap_[dimension])[index] = [value];
	}

	typename TFFunctionInterface::Ptr clone(){

		return TFFunctionInterface::Ptr(new TFRGBaFunction<dim>(*this));
	}
};*/

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_FUNCTION