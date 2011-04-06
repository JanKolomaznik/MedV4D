#ifndef TF_GRAYSCALEALPHA_PAINTER
#define TF_GRAYSCALEALPHA_PAINTER

#include <TFAbstractPainter.h>
#include <QtGui/QPixmap>

namespace M4D {
namespace GUI {

#define TF_GRAYSCALEPAINTER_DIMENSION 1

class TFGrayscaleAlphaPainter: public TFAbstractPainter<TF_GRAYSCALEPAINTER_DIMENSION>{

public:

	typedef boost::shared_ptr<TFGrayscaleAlphaPainter> Ptr;

	typedef TFWorkCopy<TF_GRAYSCALEPAINTER_DIMENSION> WorkCopy;

	TFGrayscaleAlphaPainter(bool drawAlpha);
	~TFGrayscaleAlphaPainter();

	void setArea(QRect area);
	QRect getInputArea();

	QPixmap getView(WorkCopy::Ptr workCopy);

private:

	const TF::Size colorBarSize_;
	const TF::Size margin_;
	const TF::Size spacing_;

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
	void updateHistogramView_(WorkCopy::Ptr workCopy);
	void updateGrayView_(WorkCopy::Ptr workCopy);
	void updateAlphaView_(WorkCopy::Ptr workCopy);
	void updateBottomColorBarView_(TFWorkCopy<TF_GRAYSCALEPAINTER_DIMENSION>::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALEALPHA_PAINTER