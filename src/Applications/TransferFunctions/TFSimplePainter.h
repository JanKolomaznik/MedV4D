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

	void setup(QWidget *parent, const QRect rect);

	void setup(QWidget *parent, const QRect rect, int marginH, int marginV);

	void setView(TFPointMap view);

	TFPointMap getView();

	TFPoints getPoints();

protected:
	void paintEvent(QPaintEvent *e);
	void mousePressEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);

private slots:
	void Repaint();

private:
	Ui::TFSimplePainter* _painter;

	int _marginV, _marginH;

	TFPointMap _view;	
};

#endif //TF_SIMPLEPAINTER