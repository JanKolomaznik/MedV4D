#ifndef TF_RGBA_PAINTER
#define TF_RGBA_PAINTER

#include "MedV4D/GUI/TF/Painter1D.h"

namespace M4D {
namespace GUI {

class RGBaPainter1D: public Painter1D{

public:

	typedef boost::shared_ptr<RGBaPainter1D> Ptr;

	RGBaPainter1D();
	RGBaPainter1D(
		const QColor& red,
		const QColor& green,
		const QColor& blue,
		const QColor& alpha);

	~RGBaPainter1D();
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_PAINTER
