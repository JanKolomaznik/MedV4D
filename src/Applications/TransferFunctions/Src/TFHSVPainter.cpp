#include "TFHSVPainter.h"

namespace M4D {
namespace GUI {

TFHSVPainter::TFHSVPainter():
	activeView_(ACTIVE_HUE){}

TFHSVPainter::~TFHSVPainter(){}

void TFHSVPainter::setUp(QWidget *parent){

	setParent(parent);
	correctView_();
	show();
}

void TFHSVPainter::setUp(QWidget *parent, int margin){

	setMargin_(margin);
	setUp(parent);
}

void TFHSVPainter::correctView_(){

	paintAreaWidth_ = width() - 2*margin_ - margin_ - colorBarSize_;
	paintAreaHeight_ = height() - 2*margin_	- margin_ - colorBarSize_;

	view_ = TFColorMapPtr(new TFColorMap(paintAreaWidth_));
}

void TFHSVPainter::paintEvent(QPaintEvent *e){

	QPainter drawer(this);

	QRect paintingRect(rect().left() + colorBarSize_ + margin_, rect().top(),
		rect().width() - colorBarSize_ - margin_, rect().height());

	paintSideColorBar_(&drawer);
	paintBackground_(&drawer, paintingRect, true);
	paintCurves_(&drawer);

}

void TFHSVPainter::paintSideColorBar_(QPainter *drawer){

	int yBegin = paintAreaHeight_ + margin_;
	int xBegin = 0;
	QColor color;
	for(TFSize i = 0; i < paintAreaHeight_; ++i)
	{
		color.setHsvF(i/(float)paintAreaHeight_, 1, 1);		

		drawer->setPen(color);
		drawer->drawLine(xBegin, yBegin - i, xBegin + colorBarSize_, yBegin - i);
	}
}

void TFHSVPainter::paintCurves_(QPainter *drawer){

	int xBegin = colorBarSize_ + 2*margin_;
	int yBegin = margin_ + paintAreaHeight_;
	TFPaintingPoint origin(xBegin, yBegin);

	for(TFSize i = 0; i < paintAreaWidth_ - 2; ++i)
	{
		//value
		drawer->setPen(Qt::lightGray);	
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].component3*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].component3*paintAreaHeight_);
		//saturation
		drawer->setPen(Qt::darkCyan);
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].component2*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].component2*paintAreaHeight_);
		//hue
		drawer->setPen(Qt::darkMagenta);
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].component1*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].component1*paintAreaHeight_);
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

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
	if( (mousePosition.x < (int)(colorBarSize_ + margin_))
		|| (mousePosition.y > (int)(paintAreaHeight_ + 2*margin_)) )
	{
		return;
	}
	drawHelper_ = new TFPaintingPoint(mousePosition.x - colorBarSize_ - margin_,
		(paintAreaHeight_ + 2*margin_) - mousePosition.y);
}

void TFHSVPainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFHSVPainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x() - colorBarSize_ - margin_,
		(paintAreaHeight_ + 2*margin_) - e->pos().y());

	TFPaintingPoint begin = correctCoords_(*drawHelper_);
	TFPaintingPoint end = correctCoords_(mousePosition);

	addLine_(begin, end);

	*drawHelper_ = mousePosition;
	
	if(changed()) repaint(rect());
}

void TFHSVPainter::addPoint_(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight_;
	
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
