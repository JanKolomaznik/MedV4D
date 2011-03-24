#ifndef TF_RGBA_PAINTER
#define TF_RGBA_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

#define TF_RGBPAINTER_DIMENSION 1

class TFRGBaPainter: public TFAbstractPainter<TF_RGBPAINTER_DIMENSION>{

public:

	typedef boost::shared_ptr<TFRGBaPainter> Ptr;

	TFRGBaPainter(bool drawAlpha);
	~TFRGBaPainter();

	void setArea(QRect area);
	QRect getInputArea();

	QPixmap getView(TFWorkCopy<TF_RGBPAINTER_DIMENSION>::Ptr workCopy);

private:

	const TF::Size colorBarSize_;
	const TF::Size margin_;
	const TF::Size spacing_;

	const QColor background_;
	const QColor red_;
	const QColor green_;
	const QColor blue_;
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
	QPixmap viewRedBuffer_;
	QPixmap viewGreenBuffer_;
	QPixmap viewBlueBuffer_;
	QPixmap viewAlphaBuffer_;
	QPixmap viewBottomColorBarBuffer_;

	void updateBackground_();	
	void updateHistogramView_(TFWorkCopy<TF_RGBPAINTER_DIMENSION>::Ptr workCopy);
	void updateRedView_(TFWorkCopy<TF_RGBPAINTER_DIMENSION>::Ptr workCopy);
	void updateGreenView_(TFWorkCopy<TF_RGBPAINTER_DIMENSION>::Ptr workCopy);
	void updateBlueView_(TFWorkCopy<TF_RGBPAINTER_DIMENSION>::Ptr workCopy);
	void updateAlphaView_(TFWorkCopy<TF_RGBPAINTER_DIMENSION>::Ptr workCopy);
	void updateBottomColorBarView_(TFWorkCopy<TF_RGBPAINTER_DIMENSION>::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_PAINTER