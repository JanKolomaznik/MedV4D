#ifndef TF_HSVA_PAINTER_1D
#define TF_HSVA_PAINTER_1D

#include <TFPainter1D.h>

namespace M4D {
namespace GUI {

class TFHSVaPainter1D: public TFPainter1D{

public:

	typedef boost::shared_ptr<TFHSVaPainter1D> Ptr;

	TFHSVaPainter1D();
	TFHSVaPainter1D(
		const QColor& hue,
		const QColor& saturation,
		const QColor& value,
		const QColor& alpha);

	~TFHSVaPainter1D();

	void setArea(QRect area);
	QPixmap getView(TFWorkCopy::Ptr workCopy);

private:

	QRect sideBarArea_;

	QPixmap viewSideBarBuffer_;

	void updateSideBar_(TFWorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_PAINTER_1D