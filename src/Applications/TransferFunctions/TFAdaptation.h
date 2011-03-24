#ifndef TF_ADAPTATION
#define TF_ADAPTATION

#include "GUI/utils/TransferFunctionBuffer.h"

#include <TFAbstractFunction.h>
#include <TFColor.h>

namespace M4D {
namespace GUI {
namespace TF {
namespace Adaptation {

template<typename BufferIterator>
bool applyTransferFunction(
	BufferIterator begin,
	BufferIterator end,
	TFApplyFunctionInterface::Ptr function_){

	typedef typename std::iterator_traits<BufferIterator>::value_type MultiDValueType;
	typedef typename std::iterator_traits<MultiDValueType::iterator>::value_type OneDValueType;

	TF::Size index = 0;
	TF::Color color;
	for(BufferIterator it = begin; it!=end; ++it)
	{
		if(index >= function_->getDomain())
		{
			tfAssert("Wrong buffer size");
			return false;
		}

		for(Size i = 1; i <= function_->getDimension(); ++i)
		{
			color = function_->getMappedRGBfColor(index, 1);
			
			(*it)[i] = OneDValueType(color.component1, color.component2, color.component3, color.alpha);
		}

		++index;
	}
	if(index < function_->getDomain())
	{
		tfAssert("Wrong buffer size");
		return false;
	}

	return true;
}

template<>
bool applyTransferFunction<TransferFunctionBuffer1D::iterator>(
	TransferFunctionBuffer1D::iterator begin,
	TransferFunctionBuffer1D::iterator end,
	TFApplyFunctionInterface::Ptr function_);

/*
template<typename HistogramIterator>
TF::Histogram::Ptr computeTFHistogram(HistogramIterator begin, HistogramIterator end){

	TF::Histogram::Ptr tfHistogram(new TF::Histogram);
	for(HistogramIterator it = begin; it != end; ++it){

		tfHistogram->add((TF::Size)*it);
	}

	return tfHistogram;
}
*/
}	//namespace Adaptation
}	//namespace TF
}	//namespace GUI
}	//namespace M4D

#endif	//TF_ADAPTATION