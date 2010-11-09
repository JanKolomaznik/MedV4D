#ifndef TF_ABSTRACT_PAINTER
#define TF_ABSTRACT_PAINTER

#include <QtGui/QWidget>
#include <QtGui/QMouseEvent>
#include <QtGui/QPaintEvent>

#include <QtGui/QPainter>

#include <vector>

#include <ui_TFAbstractPainter.h>

#include <TFTypes.h>

namespace M4D {
namespace GUI {

class TFAbstractPainter: public QWidget{

	Q_OBJECT

public:

	void resize(const QRect rect);

	virtual bool changed() = 0;

protected:

	Ui::TFAbstractPainter* painter_;

	TFSize margin_;
	TFSize paintAreaWidth, paintAreaHeight;
	TFPaintingPoint* drawHelper_;

	TFAbstractPainter();
	virtual ~TFAbstractPainter();

	TFPaintingPoint correctCoords(const TFPaintingPoint &point);
	TFPaintingPoint correctCoords(int x, int y);

	void addLine(int x1, int y1, int x2, int y2);
	void addLine(TFPaintingPoint begin, TFPaintingPoint end);

	virtual void paintEvent(QPaintEvent *e) = 0;
	virtual void mousePressEvent(QMouseEvent *e) = 0;
	virtual void mouseReleaseEvent(QMouseEvent *e) = 0;
	virtual void mouseMoveEvent(QMouseEvent *e) = 0;

	virtual void resize_() = 0;

	virtual void addPoint(TFPaintingPoint point) = 0;
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_PAINTER