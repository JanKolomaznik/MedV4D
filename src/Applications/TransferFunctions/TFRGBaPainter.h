#ifndef TF_RGBA_PAINTER
#define TF_RGBA_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFRGBaPainter: public TFAbstractPainter{

public:

	typedef boost::shared_ptr<TFRGBaPainter> Ptr;

	TFRGBaPainter(bool drawAlpha);
	~TFRGBaPainter();

	void setArea(QRect area);
	QRect getInputArea();

	void drawData(QPainter* drawer, TFWorkCopy::Ptr workCopy);

private:

	const TFSize colorBarSize_;
	const TFSize margin_;
	const TFSize spacing_;

	const QColor background_;
	const QColor red_;
	const QColor green_;
	const QColor blue_;
	const QColor alpha_;
	const QColor hist_;

	bool drawAlpha_;

	QRect inputArea_;
	QRect backgroundArea_;
	QRect bottomBarArea_;

	QPixmap dBuffer_;
	void drawBackground_(QPainter* drawer);
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_PAINTER