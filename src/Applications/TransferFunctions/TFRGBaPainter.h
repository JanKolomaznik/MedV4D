#ifndef TF_RGBA_PAINTER
#define TF_RGBA_PAINTER

#include <TFSimplePainter.h>

namespace M4D {
namespace GUI {

class TFRGBaPainter: public TFSimplePainter{

public:

	typedef boost::shared_ptr<TFRGBaPainter> Ptr;

	TFRGBaPainter();
	TFRGBaPainter(
		const QColor& red,
		const QColor& green,
		const QColor& blue,
		const QColor& alpha);

	~TFRGBaPainter();
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_PAINTER