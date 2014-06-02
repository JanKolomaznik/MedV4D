#ifndef TF_RGBA_FUNCTION
#define TF_RGBA_FUNCTION

#include "MedV4D/GUI/TF/AbstractFunction.h"

namespace M4D {
namespace GUI {

template<TF::Size dim>
class RGBaFunction: public AbstractFunction<dim>{

public:

	RGBaFunction(std::vector<TF::Size> domains):
		AbstractFunction<dim>(domains){
	}

	RGBaFunction(const RGBaFunction<dim> &function){

		this->colorMap_ = function.colorMap_;
	}

	void operator=(const RGBaFunction<dim> &function){

		this->colorMap_ = function.colorMap_;
	}

	~RGBaFunction(){}

	TF::Color getRGBfColor(const TF::Coordinates& coords){

		return this->color(coords); 
	}

	void setRGBfColor(const TF::Coordinates& coords, const TF::Color& value){

		this->color(coords) = value;
	}

	TransferFunctionInterface::Ptr clone(){

		return TransferFunctionInterface::Ptr(new RGBaFunction<dim>(*this));
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_FUNCTION
