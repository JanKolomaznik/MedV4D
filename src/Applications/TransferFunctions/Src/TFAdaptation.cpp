#include <TFAdaptation.h>

namespace M4D {
namespace GUI {
namespace TF {
namespace Adaptation{
/*
template<>
bool applyTransferFunction<TransferFunctionBuffer1D::Iterator>(
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
*/
}
}
}
}