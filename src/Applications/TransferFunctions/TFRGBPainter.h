#ifndef TF_RGB_PAINTER
#define TF_RGB_PAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFRGBPainter: public TFAbstractPainter{

public:
	TFRGBPainter();

	~TFRGBPainter();

	void setUp(QWidget *parent);
	void setUp(QWidget *parent, int margin);

	TFFunctionMapPtr getRedView();
	TFFunctionMapPtr getGreenView();
	TFFunctionMapPtr getBlueView();

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
		ACTIVE_BLUE
	};

	Ui::TFSimplePainter* painter_;

	ActiveView activeView_;
	TFFunctionMapPtr redView_, greenView_, blueView_;

	bool redChanged_, greenChanged_, blueChanged_;

	void addPoint(TFPaintingPoint point);
};

} // namespace GUI
} // namespace M4D

#endif //TF_RGB_PAINTER