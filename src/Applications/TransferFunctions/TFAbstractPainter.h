#ifndef TF_ABSTRACT_PAINTER
#define TF_ABSTRACT_PAINTER

#include <QtGui/QWidget>
#include <QtGui/QMouseEvent>
#include <QtGui/QPaintEvent>

#include <QtGui/QPainter>

#include <ui_TFAbstractPainter.h>

#include <TFTypes.h>

namespace M4D {
namespace GUI {

class TFAbstractPainter: public QWidget{

	Q_OBJECT

public:

	TFColorMapPtr getView();

	void resize(const QRect rect);

	bool changed();

protected:

	Ui::TFAbstractPainter* painter_;

	bool changed_;
	TFSize margin_;
	TFSize paintAreaWidth, paintAreaHeight;
	TFPaintingPoint* drawHelper_;

	TFColorMapPtr view_;

	TFAbstractPainter();
	virtual ~TFAbstractPainter();

	void setMargin_(TFSize margin);
	void paintBackground_(QPainter& painter);

	TFPaintingPoint correctCoords(const TFPaintingPoint &point);
	TFPaintingPoint correctCoords(int x, int y);

	void addLine(int x1, int y1, int x2, int y2);
	void addLine(TFPaintingPoint begin, TFPaintingPoint end);

	virtual void paintEvent(QPaintEvent *e) = 0;
	virtual void mousePressEvent(QMouseEvent *e) = 0;
	virtual void mouseReleaseEvent(QMouseEvent *e) = 0;
	virtual void mouseMoveEvent(QMouseEvent *e) = 0;

	virtual void addPoint(TFPaintingPoint point) = 0;
};

} // namespace GUI
} // namespace M4D

#endif //TF_ABSTRACT_PAINTER