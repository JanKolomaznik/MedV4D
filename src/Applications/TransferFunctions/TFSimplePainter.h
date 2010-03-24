#ifndef TF_SIMPLEPAINTER
#define TF_SIMPLEPAINTER

#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QPaintEvent>

#include <vector>

#include <TFSimpleFunction.h>

namespace Ui{

	class TFSimplePainter;
}

class TFSimplePainter: public QWidget{

	Q_OBJECT

public:
	TFSimplePainter();

	TFSimplePainter(int marginH, int marginV);

	~TFSimplePainter();

	void setup(QWidget *parent);

	void setup(QWidget *parent, int marginH, int marginV);

	void resize(const QRect rect);

	void setView(TFPointMap view);

	TFPointMap getView();

	TFPoints getPoints();

protected:
	void paintEvent(QPaintEvent *e);
	void mousePressEvent(QMouseEvent *e);
	void mouseReleaseEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);
/*
private slots:
	void Repaint();
*/
private:
	Ui::TFSimplePainter* _painter;

	int _marginV, _marginH;

	TFPointMap _view;	

	TFPoint* _drawHelper;

	TFPoint painterCoords(const TFPoint &point);

	TFPoint painterCoords(int x, int y);

	void addLine(int x1, int y1, int x2, int y2);
	void addPoint(TFPoint point);
};

#endif //TF_SIMPLEPAINTER