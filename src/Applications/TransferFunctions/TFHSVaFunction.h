#ifndef TF_HSVA_FUNCTION
#define TF_HSVA_FUNCTION

#include <TFAbstractFunction.h>
#include <GUI/utils/TransferFunctionBuffer.h>
#include <QtGui/QColor>

namespace M4D {
namespace GUI {

class TFHSVaFunction: public TFAbstractFunction{

public:

	TFHSVaFunction(const TF::Size domain);
	TFHSVaFunction(TFHSVaFunction &function);
	~TFHSVaFunction();

	TFAbstractFunction::Ptr clone();

	TF::Color getMappedRGBfColor(const TF::Size value);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_FUNCTION