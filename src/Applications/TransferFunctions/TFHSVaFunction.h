#ifndef TF_HSVA_FUNCTION
#define TF_HSVA_FUNCTION

#include <TFAbstractFunction.h>
#include <GUI/utils/TransferFunctionBuffer.h>
#include <QtGui/QColor>

namespace M4D {
namespace GUI {

class TFHSVaFunction: public TFAbstractFunction{

public:

	TFHSVaFunction(TFSize domain = 4096);
	TFHSVaFunction(TFHSVaFunction &function);

	~TFHSVaFunction();

	template<typename ElementIterator>
	bool apply(
		ElementIterator begin,
		ElementIterator end){

		tfAssert((end-begin)==domain_);
			
		tfAbort("unsupported buffer type");

		return false;
	}

	template<>
	bool apply<TransferFunctionBuffer1D::Iterator>(
		TransferFunctionBuffer1D::Iterator begin,
		TransferFunctionBuffer1D::Iterator end){
			
		typedef TransferFunctionBuffer1D::ValueType ValueType;

		TFColorMapIt currentColor = colorMap_->begin();
		for(TransferFunctionBuffer1D::Iterator it = begin; it!=end; ++it)
		{
			QColor color;
			color.setHsvF(
				currentColor->component1,
				currentColor->component2,
				currentColor->component3,
				currentColor->alpha);
			
			*it = ValueType(color.redF(), color.greenF(), color.blueF(), color.alphaF());
			++currentColor;
		}

		return true;
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_FUNCTION