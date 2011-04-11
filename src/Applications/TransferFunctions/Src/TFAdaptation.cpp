#include <TFAdaptation.h>

namespace M4D {
namespace GUI {
namespace TF {
namespace Adaptation{

template<>
bool applyTransferFunction<TransferFunctionBuffer1D::iterator>(
	TransferFunctionBuffer1D::iterator begin,
	TransferFunctionBuffer1D::iterator end,
	TFApplyFunctionInterface::Ptr function_){

	TF::Size index = 0;
	TF::Color color;
	for(TransferFunctionBuffer1D::iterator it = begin; it!=end; ++it)
	{
		if(index >= function_->getDomain())
		{
			tfAssert("Wrong buffer size");
			return false;
		}

		color = function_->getMappedRGBfColor(index, 1);

		*it = TransferFunctionBuffer1D::value_type(
			color.component1,
			color.component2,
			color.component3,
			color.alpha);

		++index;
	}
	if(index < function_->getDomain())
	{
		tfAssert("Wrong buffer size");
		return false;
	}

	return true;
}
}
}
}
}