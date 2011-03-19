#ifndef TF_RGBA_FUNCTION
#define TF_RGBA_FUNCTION

#include <TFAbstractFunction.h>
#include <GUI/utils/TransferFunctionBuffer.h>

namespace M4D {
namespace GUI {

class TFRGBaFunction: public TFAbstractFunction{

public:

	TFRGBaFunction(const TF::Size domain);
	TFRGBaFunction(TFRGBaFunction &function);
	~TFRGBaFunction();

	TFAbstractFunction::Ptr clone();

	TF::Color getMappedRGBfColor(const TF::Size value);
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_FUNCTION