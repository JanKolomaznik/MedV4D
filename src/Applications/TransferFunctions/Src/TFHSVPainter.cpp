#include "TFHSVPainter.h"

namespace M4D {
namespace GUI {

TFHSVPainter::TFHSVPainter():
	activeView_(ACTIVE_HUE){}

TFHSVPainter::~TFHSVPainter(){}

void TFHSVPainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFHSVPainter::setUp(QWidget *parent, int margin){

	setMargin_(margin);
	setUp(parent);
}

void TFHSVPainter::paintEvent(QPaintEvent *){

	QPainter painter(this);
	painter.fillRect(rect(), QBrush(Qt::black));

	int beginX = margin_;
	int beginY = height() - margin_;
	TFPaintingPoint origin(beginX, beginY);

	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		//value
		painter.setPen(/*brown*/QColor(139,69,19));	
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].component3*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].component3*paintAreaHeight);
		//saturation
		painter.setPen(Qt::lightGray);
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].component2*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].component2*paintAreaHeight);
		//hue
		painter.setPen(/*pink*/QColor(255,20,147));
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].component1*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].component1*paintAreaHeight);
	}
}

void TFHSVPainter::mousePressEvent(QMouseEvent *e){

	if(e->button() == Qt::RightButton)
	{
		switch(activeView_)
		{
			case ACTIVE_HUE:
			{
				activeView_ = ACTIVE_SATURATION;
				break;
			}
			case ACTIVE_SATURATION:
			{
				activeView_ = ACTIVE_VALUE;
				break;
			}
			case ACTIVE_VALUE:
			{
				activeView_ = ACTIVE_HUE;
				break;
			}
		}
		return;
	}

	drawHelper_ = new TFPaintingPoint(e->pos().x(), e->pos().y());
	mouseMoveEvent(e);
}

void TFHSVPainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFHSVPainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
		
	TFPaintingPoint begin = correctCoords(*drawHelper_);
	TFPaintingPoint end = correctCoords(mousePosition);

	addLine(begin, end);

	*drawHelper_ = mousePosition;
	
	if(changed()) repaint(rect());
}

void TFHSVPainter::addPoint(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight;
	
	switch(activeView_)
	{
		case ACTIVE_HUE:
		{
			(*view_)[point.x].component1 = yValue;
			break;
		}
		case ACTIVE_SATURATION:
		{
			(*view_)[point.x].component2 = yValue;
			break;
		}
		case ACTIVE_VALUE:
		{
			(*view_)[point.x].component3 = yValue;
			break;
		}
	}
	changed_ = true;
}

} // namespace GUI
} // namespace M4D
