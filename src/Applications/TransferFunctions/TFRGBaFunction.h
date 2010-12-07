#ifndef TF_RGBA_FUNCTION
#define TF_RGBA_FUNCTION

#include <TFAbstractFunction.h>
#include <GUI/utils/TransferFunctionBuffer.h>

namespace M4D {
namespace GUI {

class TFRGBaFunction: public TFAbstractFunction{

public:

	TFRGBaFunction(TFSize domain = 4096);
	TFRGBaFunction(TFRGBaFunction &function);

	~TFRGBaFunction();

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
			*it = ValueType(
				currentColor->component1, 
				currentColor->component2,
				currentColor->component3,
				currentColor->alpha);

			++currentColor;
		}

		return true;
	}
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_FUNCTION