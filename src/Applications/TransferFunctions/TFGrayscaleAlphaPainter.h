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

	QPixmap getView(TFWorkCopy::Ptr workCopy);

private:

	const TFSize colorBarSize_;
	const TFSize margin_;
	const TFSize spacing_;

	const QColor background_;
	const QColor gray_;
	const QColor alpha_;
	const QColor hist_;
	const QColor noColor_;

	bool drawAlpha_;
	bool sizeChanged_;

	QRect inputArea_;
	QRect backgroundArea_;
	QRect bottomBarArea_;

	QPixmap viewBuffer_;
	QPixmap viewBackgroundBuffer_;
	QPixmap viewHistogramBuffer_;
	QPixmap viewGrayBuffer_;
	QPixmap viewAlphaBuffer_;
	QPixmap viewBottomColorBarBuffer_;

	void updateBackground_();	
	void updateHistogramView_(TFWorkCopy::Ptr workCopy);
	void updateGrayView_(TFWorkCopy::Ptr workCopy);
	void updateAlphaView_(TFWorkCopy::Ptr workCopy);
	void updateBottomColorBarView_(TFWorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALEALPHA_PAINTER