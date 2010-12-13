#ifndef TF_RGBA_PAINTER
#define TF_RGBA_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFRGBaPainter: public TFAbstractPainter{

public:

	TFRGBaPainter(QWidget* parent);
	~TFRGBaPainter();

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
		ACTIVE_BLUE,
		ACTIVE_ALPHA
	};

	ActiveView activeView_;
	
	void paintCurves_(QPainter *drawer);
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_PAINTER