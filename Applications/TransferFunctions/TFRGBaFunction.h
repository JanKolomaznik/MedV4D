#ifndef TF_RGBA_FUNCTION
#define TF_RGBA_FUNCTION

#include <TFAbstractFunction.h>

namespace M4D {
namespace GUI {

template<TF::Size dim>
class TFRGBaFunction: public TFAbstractFunction<dim>{

public:

	TFRGBaFunction(std::vector<TF::Size> domains):
		typename TFAbstractFunction<dim>(domains){
	}

	TFRGBaFunction(const TFRGBaFunction<dim> &function){

		colorMap_ = function.colorMap_;
	}

	void operator=(const TFRGBaFunction<dim> &function){

		colorMap_ = function.colorMap_;
	}

	~TFRGBaFunction(){}

	TF::Color getRGBfColor(const TF::Coordinates& coords){

		return color(coords); 
	}

	void setRGBfColor(const TF::Coordinates& coords, const TF::Color& value){

		color(coords) = value;
	}

	TFFunctionInterface::Ptr clone(){

		return TFFunctionInterface::Ptr(new TFRGBaFunction<dim>(*this));
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_FUNCTION