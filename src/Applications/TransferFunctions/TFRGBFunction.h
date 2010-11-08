#ifndef TF_RGB_FUNCTION
#define TF_RGB_FUNCTION

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

class TFRGBFunction: public TFAbstractFunction{

public:

	TFRGBFunction(TFSize domain = 4096);
	TFRGBFunction(TFRGBFunction &function);

	~TFRGBFunction();

	void operator=(TFRGBFunction &function);
	TFAbstractFunction* clone();

	TFFunctionMapPtr getRedFunction();
	TFFunctionMapPtr getGreenFunction();
	TFFunctionMapPtr getBlueFunction();
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

		TFFunctionMapIt currentRed = red_->begin();
		TFFunctionMapIt currentGreen = green_->begin();
		TFFunctionMapIt currentBlue = blue_->begin();
		for(TransferFunctionBuffer1D::Iterator it = begin; it!=end; ++it)
		{
			*it = ValueType(*currentRed, *currentGreen, *currentBlue, 1);
			++currentRed;
			++currentGreen;
			++currentBlue;
		}

		return true;
	}

private:	
	TFFunctionMapPtr red_;
	TFFunctionMapPtr green_;
	TFFunctionMapPtr blue_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGB_FUNCTION