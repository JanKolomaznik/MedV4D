#ifndef TF_HSVA_PAINTER
#define TF_HSVA_PAINTER

#include <TFSimplePainter.h>

namespace M4D {
namespace GUI {

class TFHSVaPainter: public TFSimplePainter{

public:

	typedef boost::shared_ptr<TFHSVaPainter> Ptr;

	typedef TFSimplePainter::WorkCopy WorkCopy;

	TFHSVaPainter(bool drawAlpha);
	~TFHSVaPainter();

	void setArea(QRect area);
	QPixmap getView(WorkCopy::Ptr workCopy);

private:

	QRect sideBarArea_;

	QPixmap viewSideBarBuffer_;

	void updateSideBar_(WorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_PAINTER