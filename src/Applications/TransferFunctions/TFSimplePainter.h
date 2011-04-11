#ifndef TF_SIMPLE_PAINTER
#define TF_SIMPLE_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFSimplePainter: public TFAbstractPainter<TF_DIMENSION_1>{

public:

	typedef boost::shared_ptr<TFSimplePainter> Ptr;

	typedef TFWorkCopy<TF_DIMENSION_1> WorkCopy;

	TFSimplePainter(const QColor& component1);
	TFSimplePainter(const QColor& component1, const QColor& alpha);
	TFSimplePainter(const QColor& component1, const QColor& component2, const QColor& component3);
	TFSimplePainter(const QColor& component1, const QColor& component2, const QColor& component3, const QColor& alpha);	
	~TFSimplePainter();

	virtual void setArea(QRect area);
	QRect getInputArea();

	virtual QPixmap getView(WorkCopy::Ptr workCopy);

protected:

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
	bool firstOnly_;
	bool sizeChanged_;

	QRect inputArea_;
	QRect backgroundArea_;
	QRect bottomBarArea_;

	QPixmap viewBuffer_;
	QPixmap viewBackgroundBuffer_;
	QPixmap viewHistogramBuffer_;
	QPixmap viewComponent1Buffer_;
	QPixmap viewComponent2Buffer_;
	QPixmap viewComponent3Buffer_;
	QPixmap viewAlphaBuffer_;
	QPixmap viewBottomColorBarBuffer_;

	void updateBackground_();	
	void updateHistogramView_(WorkCopy::Ptr workCopy);
	void updateComponent1View_(WorkCopy::Ptr workCopy);
	void updateComponent2View_(WorkCopy::Ptr workCopy);
	void updateComponent3View_(WorkCopy::Ptr workCopy);
	void updateAlphaView_(WorkCopy::Ptr workCopy);
	void updateBottomColorBarView_(WorkCopy::Ptr workCopy);
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLE_PAINTER