#ifndef TF_HSVA_PAINTER
#define TF_HSVA_PAINTER

#include <TFSimplePainter.h>

namespace M4D {
namespace GUI {

class TFHSVaPainter: public TFSimplePainter{

public:

	typedef boost::shared_ptr<TFHSVaPainter> Ptr;

	TFHSVaPainter();
	TFHSVaPainter(
		const QColor& hue,
		const QColor& saturation,
		const QColor& value,
		const QColor& alpha);

	~TFHSVaPainter();

	void setArea(QRect area);
	QPixmap getView(TFWorkCopy::Ptr workCopy);

private:

	QRect sideBarArea_;

	QPixmap viewSideBarBuffer_;

	void updateSideBar_(TFWorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_PAINTER