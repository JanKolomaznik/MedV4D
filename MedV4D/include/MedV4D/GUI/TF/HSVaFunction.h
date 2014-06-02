#ifndef TF_HSVA_FUNCTION
#define TF_HSVA_FUNCTION

#include "MedV4D/GUI/TF/AbstractFunction.h"
#include <QColor>

namespace M4D {
namespace GUI {

template<TF::Size dim>
class HSVaFunction: public AbstractFunction<dim>{

public:

	HSVaFunction(std::vector<TF::Size> domains):
		AbstractFunction<dim>(domains){
	}

	HSVaFunction(const HSVaFunction<dim> &function){

		this->colorMap_ = function.colorMap_;
	}

	void operator=(const HSVaFunction<dim> &function){

		this->colorMap_ = function.colorMap_;
	}

	~HSVaFunction(){}

	TF::Color getRGBfColor(const TF::Coordinates& coords){

		TF::Color tfColor = this->color(coords);
		QColor qColor;
		qColor.setHsvF(
			tfColor.component1,
			tfColor.component2,
			tfColor.component3,
			tfColor.alpha);

		return TF::Color(qColor.redF(), qColor.greenF(), qColor.blueF(), qColor.alphaF());
	}

	void setRGBfColor(const TF::Coordinates& coords, const TF::Color& value){

		QColor qColor;
		qColor.setRgbF(value.component1, value.component2, value.component3, value.alpha);

		this->color(coords) = TF::Color(qColor.hueF(), qColor.saturationF(), qColor.valueF(), qColor.alphaF());
	}

	TransferFunctionInterface::Ptr clone(){

		return TransferFunctionInterface::Ptr(new HSVaFunction<dim>(*this));
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_FUNCTION
