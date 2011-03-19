#ifndef TF_ADAPTATION
#define TF_ADAPTATION

#include "GUI/utils/TransferFunctionBuffer.h"

#include <TFAbstractFunction.h>
#include <TFHistogram.h>

namespace M4D {
namespace GUI {
namespace TF {
namespace Adaptation {

template<typename BufferIterator>
bool applyTransferFunction(
	BufferIterator begin,
	BufferIterator end,
	TFAbstractFunction::Ptr function_){

	tfAbort("unsupported buffer type");
	return false;
}

template<>
inline bool applyTransferFunction<TransferFunctionBuffer1D::Iterator>(
	TransferFunctionBuffer1D::Iterator begin,
	TransferFunctionBuffer1D::Iterator end,
	TFAbstractFunction::Ptr function_
	){

	TF::Size index = 0;
	for(TransferFunctionBuffer1D::Iterator it = begin; it!=end; ++it)
	{
		tfAssert(index < function_->getDomain());
		TF::Color color = function_->getMappedRGBfColor(index);
		
		*it = TransferFunctionBuffer1D::ValueType(
			color.component1, color.component2, color.component3, color.alpha);

		++index;
	}

	return true;
}

template<typename HistogramIterator>
TF::Histogram::Ptr computeTFHistogram(HistogramIterator begin, HistogramIterator end){

	TF::Histogram::Ptr tfHistogram(new TF::Histogram);
	for(HistogramIterator it = begin; it != end; ++it){

		tfHistogram->add((TF::Size)*it);
	}

	return tfHistogram;
}

}
}
}
}

#endif	//TF_ADAPTATION