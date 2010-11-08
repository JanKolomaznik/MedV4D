#ifndef TF_SIMPLEPAINTER
#define TF_SIMPLEPAINTER

#include <TFAbstractPainter.h>

namespace M4D {
namespace GUI {

class TFSimplePainter: public TFAbstractPainter{

public:
	TFSimplePainter();

	~TFSimplePainter();

	void setUp(QWidget *parent);
	void setUp(QWidget *parent, int margin);

	TFFunctionMapPtr getView();

	bool changed();

protected:
	void paintEvent(QPaintEvent *e);
	void mousePressEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);

	void resize_();

private:

	bool changed_;

	TFFunctionMapPtr view_;	

	void addPoint(TFPaintingPoint point);
};

} // namespace GUI
} // namespace M4D

#endif //TF_SIMPLEPAINTER