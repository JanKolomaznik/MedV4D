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

	QPixmap getView(TFWorkCopy::Ptr workCopy);

private:

	const TF::Size colorBarSize_;
	const TF::Size margin_;
	const TF::Size spacing_;

	const QColor background_;
	const QColor hue_;
	const QColor saturation_;
	const QColor value_;
	const QColor alpha_;
	const QColor hist_;
	const QColor noColor_;

	bool drawAlpha_;
	bool sizeChanged_;

	QRect inputArea_;
	QRect backgroundArea_;
	QRect bottomBarArea_;
	QRect sideBarArea_;

	QPixmap viewBuffer_;
	QPixmap viewBackgroundBuffer_;
	QPixmap viewHistogramBuffer_;
	QPixmap viewHueBuffer_;
	QPixmap viewSaturationBuffer_;
	QPixmap viewValueBuffer_;
	QPixmap viewAlphaBuffer_;
	QPixmap viewBottomColorBarBuffer_;

	void updateBackground_();	
	void updateHistogramView_(TFWorkCopy::Ptr workCopy);
	void updateHueView_(TFWorkCopy::Ptr workCopy);
	void updateSaturationView_(TFWorkCopy::Ptr workCopy);
	void updateValueView_(TFWorkCopy::Ptr workCopy);
	void updateAlphaView_(TFWorkCopy::Ptr workCopy);
	void updateBottomColorBarView_(TFWorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_PAINTER