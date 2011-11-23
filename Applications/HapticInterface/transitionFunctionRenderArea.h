#ifndef TRANSITION_FUNCTION_RENDERAREA_H
#define TRANSITION_FUNCTION_RENDERAREA_H

#include <QPen>
#include <QWidget>
#include "transitionFunction.h"

class transitionFunctionRenderAreaWidget : public QWidget
{
    Q_OBJECT

public:
    transitionFunctionRenderAreaWidget(transitionFunction* functionData, QWidget *parent = 0);

    QSize minimumSizeHint() const;
    QSize sizeHint() const;

public slots:
	void stateChangedSlot(int state);

signals:
	void addPointSignal(double a_x, double a_y);
	void mouseCoordinatesChangedSignal(double a_x, double a_y);

protected:
    void paintEvent(QPaintEvent * event);
	void mouseReleaseEvent(QMouseEvent * event);
	void mouseMoveEvent(QMouseEvent * event);
	void mousePressEvent( QMouseEvent * event );
	void leaveEvent( QEvent *event );

private:
	transitionFunction* functionData;
    QPen pen;
	bool pointStyleMoveable, deny;
	int selectedPoint;
	int lengthToPickPoint;
	int mouseX, mouseY;
};
//! [0]

#endif
