#ifndef TF_GA_PAINTER_1D
#define TF_GA_PAINTER_1D

#include "MedV4D/GUI/TF/Painter1D.h"

namespace M4D {
namespace GUI {

class GrayscaleAlphaPainter1D: public Painter1D{

public:

	typedef boost::shared_ptr<GrayscaleAlphaPainter1D> Ptr;

	GrayscaleAlphaPainter1D();
	GrayscaleAlphaPainter1D(
		const QColor& gray,
		const QColor& alpha);

	~GrayscaleAlphaPainter1D();

	void updateFunctionView_(WorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_GA_PAINTER_1D
