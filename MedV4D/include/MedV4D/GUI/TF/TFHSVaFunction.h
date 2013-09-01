#ifndef TF_HSVA_FUNCTION
#define TF_HSVA_FUNCTION

#include "MedV4D/GUI/TF/TFAbstractFunction.h"
#include <QColor>

namespace M4D {
namespace GUI {

template<TF::Size dim>
class TFHSVaFunction: public TFAbstractFunction<dim>{

public:

	TFHSVaFunction(std::vector<TF::Size> domains):
		TFAbstractFunction<dim>(domains){
	}

	TFHSVaFunction(const TFHSVaFunction<dim> &function){

		this->colorMap_ = function.colorMap_;
	}

	void operator=(const TFHSVaFunction<dim> &function){

		this->colorMap_ = function.colorMap_;
	}

	~TFHSVaFunction(){}

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

	TFFunctionInterface::Ptr clone(){

		return TFFunctionInterface::Ptr(new TFHSVaFunction<dim>(*this));
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_FUNCTION
