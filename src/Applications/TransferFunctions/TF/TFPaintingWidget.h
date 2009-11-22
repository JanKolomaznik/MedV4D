#ifndef TF_PAINTINGWIDGET
#define TF_PAINTINGWIDGET

#include <QtGui/QWidget>
#include <QtGui/QPainter>
#include <QtGui/QMouseEvent>

#include <vector>

#include <TF/TFScheme.h>

class PaintingWidget: public QWidget{
public:
	PaintingWidget(QWidget *parent){
		setParent(parent);
		_marginV = _marginH = 10;
	}

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