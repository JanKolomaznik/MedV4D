#ifndef TF_GRAYSCALETRANSPARENCY_PAINTER
#define TF_GRAYSCALETRANSPARENCY_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFGrayscaleTransparencyPainter: public TFAbstractPainter{

public:
	TFGrayscaleTransparencyPainter();

	~TFGrayscaleTransparencyPainter();

	void setUp(QWidget *parent);
	void setUp(QWidget *parent, int margin);

	TFFunctionMapPtr getGrayscaleView();
	TFFunctionMapPtr getTransparencyView();

	bool changed();

protected:
	void paintEvent(QPaintEvent *e);
	void mousePressEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);

	void resize_();

private:

	enum ActiveView{
		ACTIVE_GRAYSCALE,
		ACTIVE_TRANSPARENCY
	};

	Ui::TFSimplePainter* painter_;

	ActiveView activeView_;
	TFFunctionMapPtr grayView_, transparencyView_;

	bool grayChanged_, transparencyChanged_;

	void addPoint(TFPaintingPoint point);
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALETRANSPARENCY_PAINTER