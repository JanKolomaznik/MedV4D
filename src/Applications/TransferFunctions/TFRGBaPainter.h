#ifndef TF_RGBA_PAINTER
#define TF_RGBA_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFRGBaPainter: public TFAbstractPainter{

public:
	TFRGBaPainter();

	~TFRGBaPainter();

	void setUp(QWidget *parent);
	void setUp(QWidget *parent, int margin);

	TFFunctionMapPtr getRedView();
	TFFunctionMapPtr getGreenView();
	TFFunctionMapPtr getBlueView();
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
		ACTIVE_RED,
		ACTIVE_GREEN,
		ACTIVE_BLUE,
		ACTIVE_TRANSPARENCY
	};

	ActiveView activeView_;
	TFFunctionMapPtr redView_, greenView_, blueView_, transparencyView_;

	bool redChanged_, greenChanged_, blueChanged_, transparencyChanged_;

	void addPoint(TFPaintingPoint point);
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGBA_PAINTER