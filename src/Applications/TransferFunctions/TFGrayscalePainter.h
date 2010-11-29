#ifndef TF_GRAYSCALE_PAINTER
#define TF_GRAYSCALE_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFGrayscalePainter: public TFAbstractPainter{

public:

	TFGrayscalePainter();
	~TFGrayscalePainter();

	void setUp(QWidget *parent);
	void setUp(QWidget *parent, int margin);

protected:
	void paintEvent(QPaintEvent *e);
	void mousePressEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);

	void addPoint_(TFPaintingPoint point);
	void correctView_();

private:
	
	void paintCurve_(QPainter *drawer);
};

} // namespace GUI
} // namespace M4D

#endif //TF_GRAYSCALE_PAINTER