
#include "TFPaintingWidget.h"

#include <cassert>

void PaintingWidget::setView(TFFunction** function){

	currentView = function;
}

void PaintingWidget::paintEvent(QPaintEvent *){

	QPainter painter(this);
	painter.setPen(Qt::white);
	painter.fillRect(rect(), QBrush(Qt::black));
	
	TFPointsIterator first = (*currentView)->begin();
	TFPointsIterator end = (*currentView)->end();	
	TFPointsIterator it = first;

	TFPoint origin = TFPoint(_marginH, this->height() - _marginV);

	TFPoint point1 = origin;
	TFPoint point2;

	for(it; it != end; ++it)
	{
		point2 = TFPoint(origin.x + it->second->x, origin.y - it->second->y);

		painter.drawLine(point1.x, point1.y, point2.x, point2.y);
		point1 = point2;
	}
	point2 = TFPoint(this->width() - 10, origin.y);
	painter.drawLine(point1.x, point1.y, point2.x, point2.y);
}

void PaintingWidget::mouseMoveEvent(QMouseEvent *e){

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