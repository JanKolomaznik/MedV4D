#ifndef TF_GRAYSCALEALPHA_PAINTER
#define TF_GRAYSCALEALPHA_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFGrayscaleAlphaPainter: public TFAbstractPainter{

public:

	typedef boost::shared_ptr<TFGrayscaleAlphaPainter> Ptr;

	TFGrayscaleAlphaPainter(bool drawAlpha);
	~TFGrayscaleAlphaPainter();

	void setArea(TFArea area);
	const TFArea& getInputArea();

	void drawBackground(QPainter* drawer);
	void drawData(QPainter* drawer, TFWorkCopy::Ptr workCopy);

private:

	const TFSize colorBarSize_;
	const TFSize margin_;
	const TFSize spacing_;

	const QColor background_;
	const QColor grey_;
	const QColor alpha_;

	bool drawAlpha_;

	TFArea inputArea_;
	TFArea backgroundArea_;
	TFArea bottomBarArea_;
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALEALPHA_PAINTER