#ifndef TF_HSVA_PAINTER
#define TF_HSVA_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFHSVaPainter: public TFAbstractPainter{

public:

	typedef boost::shared_ptr<TFHSVaPainter> Ptr;

	TFHSVaPainter(bool drawAlpha);
	~TFHSVaPainter();

	void setArea(QRect area);
	QRect getInputArea();

	void drawData(QPainter* drawer, TFWorkCopy::Ptr workCopy);

private:

	const TFSize colorBarSize_;
	const TFSize margin_;
	const TFSize spacing_;

	const QColor background_;
	const QColor hue_;
	const QColor saturation_;
	const QColor value_;
	const QColor alpha_;
	const QColor hist_;

	bool drawAlpha_;

	QRect inputArea_;
	QRect backgroundArea_;
	QRect bottomBarArea_;
	QRect sideBarArea_;

	QPixmap dBuffer_;
	void drawBackground_(QPainter* drawer);
	void drawSideColorBar_(QPainter* drawer);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_PAINTER