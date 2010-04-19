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

protected:
    void paintEvent(QPaintEvent *event);

private:
	transitionFunction* functionData;
    QPen pen;
};
//! [0]

#endif
