#include "TFRGBaPainter.h"

namespace M4D {
namespace GUI {

TFRGBaPainter::TFRGBaPainter(QWidget* parent):
	TFAbstractPainter(parent),
	activeView_(ACTIVE_RED){
}

TFRGBaPainter::~TFRGBaPainter(){}

void TFRGBaPainter::correctView_(){

	paintAreaWidth_ = width() - 2*margin_;
	paintAreaHeight_ = height() - 2*margin_	- margin_ - colorBarSize_;

	view_ = TFColorMapPtr(new TFColorMap(paintAreaWidth_));
}

void TFRGBaPainter::paintEvent(QPaintEvent*){

	QPainter drawer(this);
	paintBackground_(&drawer, rect());
	paintCurves_(&drawer);
}

void TFRGBaPainter::paintCurves_(QPainter *drawer){

	int xBegin = margin_;
	int yBegin = margin_ + paintAreaHeight_;
	TFPaintingPoint origin(xBegin, yBegin);

	for(TFSize i = 0; i < paintAreaWidth_ - 2; ++i)
	{
		//alpha
		drawer->setPen(Qt::yellow);
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].alpha*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].alpha*paintAreaHeight_);
		//blue
		drawer->setPen(Qt::blue);	
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].component3*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].component3*paintAreaHeight_);
		//green
		drawer->setPen(Qt::green);
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].component2*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].component2*paintAreaHeight_);
		//red
		drawer->setPen(Qt::red);
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].component1*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].component1*paintAreaHeight_);
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

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
	if( mousePosition.y > (int)(paintAreaHeight_ + 2*margin_) )
	{
		return;
	}
	drawHelper_ = new TFPaintingPoint(mousePosition.x,
		(paintAreaHeight_ + 2*margin_) - mousePosition.y);
}

void TFRGBaPainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFRGBaPainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x(), (paintAreaHeight_ + 2*margin_) - e->pos().y());
		
	TFPaintingPoint begin = correctCoords_(*drawHelper_);
	TFPaintingPoint end = correctCoords_(mousePosition);

	addLine_(begin, end);

	*drawHelper_ = mousePosition;
	
	if(changed()) repaint();
}

void TFRGBaPainter::addPoint_(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight_;
	
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
