#ifndef TF_HSVA_PAINTER_1D
#define TF_HSVA_PAINTER_1D

#include "MedV4D/GUI/TF/Painter1D.h"

namespace M4D {
namespace GUI {

class HSVaPainter1D: public Painter1D{

public:

	typedef std::shared_ptr<HSVaPainter1D> Ptr;

	HSVaPainter1D();
	HSVaPainter1D(
		const QColor& hue,
		const QColor& saturation,
		const QColor& value,
		const QColor& alpha);

	~HSVaPainter1D();

	void setArea(QRect area);
	QPixmap getView(WorkCopy::Ptr workCopy);

private:

	QRect sideBarArea_;

	QPixmap viewSideBarBuffer_;

	void updateSideBar_(WorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_PAINTER_1D
