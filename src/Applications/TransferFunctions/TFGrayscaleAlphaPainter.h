#ifndef TF_GRAYSCALEALPHA_PAINTER
#define TF_GRAYSCALEALPHA_PAINTER

#include <TFAbstractPainter.h>
#include <QtGui/QPixmap>

namespace M4D {
namespace GUI {

class TFGrayscaleAlphaPainter: public TFAbstractPainter{

public:

	typedef boost::shared_ptr<TFGrayscaleAlphaPainter> Ptr;

	TFGrayscaleAlphaPainter(bool drawAlpha);
	~TFGrayscaleAlphaPainter();

	void setArea(QRect area);
	QRect getInputArea();
	void setWorkCopy(TFWorkCopy::Ptr workCopy);

	void drawData(QPainter* drawer, TFWorkCopy::Ptr workCopy);

private:

	const TFSize colorBarSize_;
	const TFSize margin_;
	const TFSize spacing_;

	const QColor background_;
	const QColor gray_;
	const QColor alpha_;

	bool drawAlpha_;

	QRect inputArea_;
	QRect backgroundArea_;
	QRect bottomBarArea_;

	QPixmap dBuffer_;
	void drawBackground_(QPainter* drawer);
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALEALPHA_PAINTER