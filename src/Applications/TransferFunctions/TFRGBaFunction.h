#ifndef TF_RGBA_FUNCTION
#define TF_RGBA_FUNCTION

#include <TFAbstractFunction.h>
#include <GUI/utils/TransferFunctionBuffer.h>

namespace M4D {
namespace GUI {

class TFRGBaFunction: public TFAbstractFunction{

public:

	TFRGBaFunction(const TFSize& domain);
	TFRGBaFunction(TFRGBaFunction &function);

	~TFRGBaFunction();

	TFColor getMappedRGBfColor(const TFSize& value);
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_FUNCTION