#ifndef TF_SIMPLE_PAINTER
#define TF_SIMPLE_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFSimplePainter: public TFAbstractPainter{

public:

	typedef boost::shared_ptr<TFSimplePainter> Ptr;
	~TFSimplePainter();

	virtual std::vector<std::string> getComponentNames();

	virtual void setArea(QRect area);
	QRect getInputArea();

	virtual QPixmap getView(TFWorkCopy::Ptr workCopy);

protected:

	TFSimplePainter(
		const QColor& component1,
		const QColor& component2,
		const QColor& component3,
		const QColor& alpha = QColor(255,127,0));	

	std::vector<std::string> componentNames_;

	const TF::Size colorBarSize_;
	const TF::Size margin_;
	const TF::Size spacing_;

	const QColor background_;
	const QColor component1_;
	const QColor component2_;
	const QColor component3_;
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
	QPixmap viewFunctionBuffer_;
	QPixmap viewBottomColorBarBuffer_;

	void updateBackground_();	
	void updateHistogramView_(TFWorkCopy::Ptr workCopy);
	virtual void updateFunctionView_(TFWorkCopy::Ptr workCopy);
	void updateBottomColorBarView_(TFWorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLE_PAINTER