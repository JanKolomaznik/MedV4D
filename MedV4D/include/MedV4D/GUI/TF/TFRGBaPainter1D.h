#ifndef TF_RGBA_PAINTER
#define TF_RGBA_PAINTER

#include "MedV4D/GUI/TF/TFPainter1D.h"

namespace M4D {
namespace GUI {

class TFRGBaPainter1D: public TFPainter1D{

public:

	typedef boost::shared_ptr<TFRGBaPainter1D> Ptr;

	TFRGBaPainter1D();
	TFRGBaPainter1D(
		const QColor& red,
		const QColor& green,
		const QColor& blue,
		const QColor& alpha);

	~TFRGBaPainter1D();
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_PAINTER
