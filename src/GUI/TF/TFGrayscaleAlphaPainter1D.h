#ifndef TF_GA_PAINTER_1D
#define TF_GA_PAINTER_1D

#include "GUI/TF/TFPainter1D.h"

namespace M4D {
namespace GUI {

class TFGrayscaleAlphaPainter1D: public TFPainter1D{

public:

	typedef boost::shared_ptr<TFGrayscaleAlphaPainter1D> Ptr;

	TFGrayscaleAlphaPainter1D();
	TFGrayscaleAlphaPainter1D(
		const QColor& gray,
		const QColor& alpha);

	~TFGrayscaleAlphaPainter1D();

	void updateFunctionView_(TFWorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_GA_PAINTER_1D
