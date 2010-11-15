#ifndef TF_GRAYSCALE_FUNCTION
#define TF_GRAYSCALE_FUNCTION

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

class TFGrayscaleFunction: public TFAbstractFunction{

public:

	TFGrayscaleFunction(TFSize domain = 4096);
	TFGrayscaleFunction(TFGrayscaleFunction &function);

	~TFGrayscaleFunction();

	void operator=(TFGrayscaleFunction &function);
	TFAbstractFunction* clone();

	void setPoint(TFSize point, float value);
	void setFunction(TFColorRGBaMapPtr function);

	TFColorRGBaMapPtr getFunction();
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

		tfAssert((end-begin)==points_->size());
			
		typedef TransferFunctionBuffer1D::ValueType ValueType;

		TFColorRGBaMapIt currentPoint = points_->begin();
		for(TransferFunctionBuffer1D::Iterator it = begin; it!=end; ++it)
		{
			float value = *currentPoint;
			*it = ValueType(value, value, value, 1);
			++currentPoint;
		}

		return true;
	}

private:	
	TFColorRGBaMapPtr points_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALE_FUNCTION