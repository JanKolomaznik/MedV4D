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

	void setArea(TFArea area);
	const TFArea& getInputArea();

	void drawBackground(QPainter* drawer);
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

	bool drawAlpha_;

	TFArea inputArea_;
	TFArea backgroundArea_;
	TFArea sideBarArea_;
	TFArea bottomBarArea_;

	void drawSideColorBar_(QPainter* drawer);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_PAINTER