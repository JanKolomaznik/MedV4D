
#include "TFPaintingWidget.h"

#include <cassert>

void TFPaintingWidget::setView(TFFunction** function){

	currentView = function;
}

void TFPaintingWidget::paintEvent(QPaintEvent *){

	QPainter painter(this);
	painter.setPen(Qt::white);
	painter.fillRect(rect(), QBrush(Qt::black));
	
	vector<TFPoint*> points = (*currentView)->getAllPoints();
	vector<TFPoint*>::iterator first = points.begin();
	vector<TFPoint*>::iterator end = points.end();
	vector<TFPoint*>::iterator it = first;

	TFPoint origin = TFPoint(_marginH, this->height() - _marginV);
	TFPoint* point1 = new TFPoint();

	for(it; it != end; ++it)
	{
		painter.drawLine(origin.x + point1->x, origin.y - point1->y, origin.x + (*it)->x, origin.y - (*it)->y);
		delete point1;
		point1 = *it;
	}

	painter.drawLine(origin.x + point1->x, origin.y - point1->y, this->width() - _marginH, origin.y);
	delete point1;
}

void TFPaintingWidget::mousePressEvent(QMouseEvent *e){
	mouseMoveEvent(e);
}

void TFPaintingWidget::mouseMoveEvent(QMouseEvent *e){

	TFPoint size = TFPoint(this->width(), this->height());
	TFPoint mousePosition = TFPoint(e->pos().x(), e->pos().y());

	if( mousePosition.x < _marginH )
	{
		mousePosition.x = _marginH;
	}

	if( mousePosition.x > (size.x - _marginH) )
	{
		mousePosition.x = size.x - _marginH;
	}

	if( mousePosition.y < _marginV )
	{
		mousePosition.y = _marginV;
	}

	if( mousePosition.y > (size.y - _marginV) )
	{
		mousePosition.y = size.y - _marginV;
	}

	assert( mousePosition.x >= _marginH &&
		mousePosition.x <= (size.x - _marginH) &&
	    mousePosition.y >= _marginV &&
	    mousePosition.y <= (size.y - _marginV) );
		
	(*currentView)->addPoint(
			new TFPoint(mousePosition.x - _marginH, size.y - mousePosition.y - _marginV));

	repaint();
}
