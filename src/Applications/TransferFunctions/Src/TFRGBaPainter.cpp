#include "TFRGBaPainter.h"

namespace M4D {
namespace GUI {

TFRGBaPainter::TFRGBaPainter():
	activeView_(ACTIVE_RED){}

TFRGBaPainter::~TFRGBaPainter(){}

void TFRGBaPainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFRGBaPainter::setUp(QWidget *parent, int margin){

	setMargin_(margin);
	setUp(parent);
}

void TFRGBaPainter::paintEvent(QPaintEvent *){

	QPainter painter(this);
	painter.fillRect(rect(), QBrush(Qt::black));

	int beginX = margin_;
	int beginY = height() - margin_;
	TFPaintingPoint origin(beginX, beginY);

	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		//alpha
		painter.setPen(Qt::yellow);
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].alpha*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].alpha*paintAreaHeight);
		//blue
		painter.setPen(Qt::blue);	
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].component3*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].component3*paintAreaHeight);
		//green
		painter.setPen(Qt::green);
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].component2*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].component2*paintAreaHeight);
		//red
		painter.setPen(Qt::red);
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].component1*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].component1*paintAreaHeight);
	}
}

void TFRGBaPainter::mousePressEvent(QMouseEvent *e){

	if(e->button() == Qt::RightButton)
	{
		switch(activeView_)
		{
			case ACTIVE_RED:
			{
				activeView_ = ACTIVE_GREEN;
				break;
			}
			case ACTIVE_GREEN:
			{
				activeView_ = ACTIVE_BLUE;
				break;
			}
			case ACTIVE_BLUE:
			{
				activeView_ = ACTIVE_ALPHA;
				break;
			}
			case ACTIVE_ALPHA:
			{
				activeView_ = ACTIVE_RED;
				break;
			}
		}
		return;
	}

	drawHelper_ = new TFPaintingPoint(e->pos().x(), e->pos().y());
	mouseMoveEvent(e);
}

void TFRGBaPainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFRGBaPainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
		
	TFPaintingPoint begin = correctCoords(*drawHelper_);
	TFPaintingPoint end = correctCoords(mousePosition);

	addLine(begin, end);

	*drawHelper_ = mousePosition;
	
	if(changed()) repaint(rect());
}

void TFRGBaPainter::addPoint(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight;
	
	switch(activeView_)
	{
		case ACTIVE_RED:
		{
			(*view_)[point.x].component1 = yValue;
			break;
		}
		case ACTIVE_GREEN:
		{
			(*view_)[point.x].component2 = yValue;
			break;
		}
		case ACTIVE_BLUE:
		{
			(*view_)[point.x].component3 = yValue;
			break;
		}
		case ACTIVE_ALPHA:
		{
			(*view_)[point.x].alpha = yValue;
			break;
		}
	}
	changed_ = true;
}

} // namespace GUI
} // namespace M4D
