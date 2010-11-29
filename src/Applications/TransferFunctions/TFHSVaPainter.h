#ifndef TF_HSVA_PAINTER
#define TF_HSVA_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFHSVaPainter: public TFAbstractPainter{

public:
	TFHSVaPainter();
	~TFHSVaPainter();

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

	enum ActiveView{
		ACTIVE_HUE,
		ACTIVE_SATURATION,
		ACTIVE_VALUE,
		ACTIVE_ALPHA
	};
	
	ActiveView activeView_;

	void paintSideColorBar_(QPainter* drawer);
	void paintCurves_(QPainter* drawer);
};

} // namespace GUI
} // namespace M4D

#endif //TF_HSVA_PAINTER