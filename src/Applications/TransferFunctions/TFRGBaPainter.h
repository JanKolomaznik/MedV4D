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

	void setArea(TFArea area);
	TFArea getInputArea();

	void drawBackground(QPainter* drawer);
	void drawData(QPainter* drawer, TFColorMapPtr workCopy);

private:

	const TFSize colorBarSize_;
	const TFSize margin_;
	const TFSize spacing_;

	const QColor background_;
	const QColor red_;
	const QColor green_;
	const QColor blue_;
	const QColor alpha_;

	bool drawAlpha_;

	TFArea inputArea_;
	TFArea backgroundArea_;
	TFArea bottomBarArea_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_PAINTER