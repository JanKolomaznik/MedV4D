#ifndef TF_PAINTINGWIDGET
#define TF_PAINTINGWIDGET

#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>
#include <QtGui/QPaintEvent>

#include <vector>

#include <TF/TFFunction.h>

class TFPaintingWidget: public QWidget{

public:
	inline TFPaintingWidget(int marginH, int marginV): _marginH(marginH), _marginV(marginV){}

	void setView(TFFunction** function);

protected:
	void paintEvent(QPaintEvent *e);
	void mousePressEvent(QMouseEvent *e);
	void mouseMoveEvent(QMouseEvent *e);

private:
	int _marginV, _marginH;
	TFFunction** currentView;
};

#endif //TF_PAINTINGWIDGET