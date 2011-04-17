#ifndef TF_HSVA_FUNCTION
#define TF_HSVA_FUNCTION

#include <TFAbstractFunction.h>
#include <QtGui/QColor>

namespace M4D {
namespace GUI {

template<TF::Size dim>
class TFHSVaFunction: public TFAbstractFunction<dim>{

public:

	TFHSVaFunction(const std::vector<TF::Size>& domains):
		typename TFAbstractFunction<dim>(domains){
	}

	TFHSVaFunction(TFHSVaFunction<dim> &function){

		operator=(function);
	}

	~TFHSVaFunction(){}

	TF::Color getRGBfColor(const TF::Size dimension, const TF::Size index){

		TF::Size innerDimension = dimension - 1;
		QColor color;
		color.setHsvF(
			(*colorMap_[innerDimension])[index].component1,
			(*colorMap_[innerDimension])[index].component2,
			(*colorMap_[innerDimension])[index].component3,
			(*colorMap_[innerDimension])[index].alpha);
		
		return TF::Color(color.redF(), color.greenF(), color.blueF(), color.alphaF());
	}

	void setRGBfColor(const TF::Size dimension, const TF::Size index, const TF::Color& value){
		
		QColor color;
		color.setRgbF(value.component1, value.component2, value.component3, value.alpha);

		TF::Size innerDimension = dimension - 1;
		(*colorMap_[innerDimension])[index].component1 = color.hueF();
		(*colorMap_[innerDimension])[index].component2 = color.saturationF();
		(*colorMap_[innerDimension])[index].component3 = color.valueF();
		(*colorMap_[innerDimension])[index].alpha = color.alphaF();
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_FUNCTION