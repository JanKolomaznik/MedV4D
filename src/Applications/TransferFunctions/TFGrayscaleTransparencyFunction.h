#ifndef TF_GRAYSCALETRANSPARENCY_FUNCTION
#define TF_GRAYSCALETRANSPARENCY_FUNCTION

#include <map>
#include <vector>
#include <algorithm>
#include <string>
#include <fstream>
#include <cassert>

#include <TFAbstractFunction.h>
#include <GUI/utils/TransferFunctionBuffer.h>

namespace M4D {
namespace GUI {

class TFGrayscaleTransparencyFunction: public TFAbstractFunction{

public:

	TFGrayscaleTransparencyFunction(TFSize domain = 4096);
	TFGrayscaleTransparencyFunction(TFGrayscaleTransparencyFunction &function);

	~TFGrayscaleTransparencyFunction();

	void operator=(TFGrayscaleTransparencyFunction &function);
	TFAbstractFunction* clone();

	TFColorRGBaMapPtr getGrayscaleFunction();
	TFColorRGBaMapPtr getTransparencyFunction();
	TFSize getDomain();

	void clear();

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

		TFColorRGBaMapIt currentGray = gray_->begin();
		TFColorRGBaMapIt currentTransparency = transparency_->begin();
		for(TransferFunctionBuffer1D::Iterator it = begin; it!=end; ++it)
		{
			*it = ValueType(*currentGray, *currentGray, *currentGray, *currentTransparency);
			++currentGray;
			++currentTransparency;
		}

		return true;
	}

private:	
	TFColorRGBaMapPtr gray_;
	TFColorRGBaMapPtr transparency_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALETRANSPARENCY_FUNCTION