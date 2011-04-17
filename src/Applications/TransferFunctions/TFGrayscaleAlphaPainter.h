#ifndef TF_GA_PAINTER
#define TF_GA_PAINTER

#include <TFSimplePainter.h>

namespace M4D {
namespace GUI {

class TFGrayscaleAlphaPainter: public TFSimplePainter{

public:

	typedef boost::shared_ptr<TFGrayscaleAlphaPainter> Ptr;

	TFGrayscaleAlphaPainter();
	TFGrayscaleAlphaPainter(
		const QColor& gray,
		const QColor& alpha);

	~TFGrayscaleAlphaPainter();

	QPixmap getView(TFWorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_GA_PAINTER