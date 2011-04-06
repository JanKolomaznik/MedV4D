#ifndef TF_RGBA_PAINTER
#define TF_RGBA_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

#define TF_RGBPAINTER_DIMENSION 1

class TFRGBaPainter: public TFAbstractPainter<TF_RGBPAINTER_DIMENSION>{

public:

	typedef boost::shared_ptr<TFRGBaPainter> Ptr;

	typedef TFWorkCopy<TF_RGBPAINTER_DIMENSION> WorkCopy;

	TFRGBaPainter(bool drawAlpha);
	~TFRGBaPainter();

	void setArea(QRect area);
	QRect getInputArea();

	QPixmap getView(WorkCopy::Ptr workCopy);

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
	void updateHistogramView_(WorkCopy::Ptr workCopy);
	void updateRedView_(WorkCopy::Ptr workCopy);
	void updateGreenView_(WorkCopy::Ptr workCopy);
	void updateBlueView_(WorkCopy::Ptr workCopy);
	void updateAlphaView_(WorkCopy::Ptr workCopy);
	void updateBottomColorBarView_(WorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_PAINTER