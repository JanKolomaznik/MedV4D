#ifndef TF_RGB_PAINTER
#define TF_RGB_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFRGBPainter: public TFAbstractPainter{

public:

	TFRGBPainter(QWidget* parent);
	~TFRGBPainter();

protected:

	void paintEvent(QPaintEvent*);
	void mousePressEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);

	void addPoint_(TFPaintingPoint point);
	void correctView_();

private:

	enum ActiveView{
		ACTIVE_RED,
		ACTIVE_GREEN,
		ACTIVE_BLUE
	};

	ActiveView activeView_;
	
	void paintCurves_(QPainter *drawer);
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGB_PAINTER