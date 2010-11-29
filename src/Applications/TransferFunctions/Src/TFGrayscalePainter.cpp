#include "TFGrayscalePainter.h"

namespace M4D {
namespace GUI {

TFGrayscalePainter::TFGrayscalePainter(){}

TFGrayscalePainter::~TFGrayscalePainter(){}

void TFGrayscalePainter::setUp(QWidget *parent){

	setParent(parent);
	correctView_();
	show();
}

void TFGrayscalePainter::setUp(QWidget *parent, int margin){

	setMargin_(margin);
	setUp(parent);
}

void TFGrayscalePainter::correctView_(){

	paintAreaWidth_ = width() - 2*margin_;
	paintAreaHeight_ = height() - 2*margin_	- margin_ - colorBarSize_;

	view_ = TFColorMapPtr(new TFColorMap(paintAreaWidth_));
}

void TFGrayscalePainter::paintEvent(QPaintEvent *e){

	QPainter drawer(this);
	paintBackground_(&drawer, rect());
	paintCurve_(&drawer);
}

void TFGrayscalePainter::paintCurve_(QPainter *drawer){

	int xBegin = margin_;
	int yBegin = margin_ + paintAreaHeight_;
	TFPaintingPoint origin(xBegin, yBegin);

	drawer->setPen(Qt::lightGray);
	for(TFSize i = 0; i < paintAreaWidth_ - 2; ++i)
	{
		drawer->drawLine(origin.x + i, origin.y - (*view_)[i].component1*paintAreaHeight_,
			origin.x + i + 1, origin.y - (*view_)[i+1].component1*paintAreaHeight_);
	}
}

void TFGrayscalePainter::mousePressEvent(QMouseEvent *e){

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
	if( mousePosition.y > (int)(paintAreaHeight_ + 2*margin_) )
	{
		return;
	}
	drawHelper_ = new TFPaintingPoint(mousePosition.x,
		(paintAreaHeight_ + 2*margin_) - mousePosition.y);
}

void TFGrayscalePainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFGrayscalePainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x(), (paintAreaHeight_ + 2*margin_) - e->pos().y());
		
	TFPaintingPoint begin = correctCoords_(*drawHelper_);
	TFPaintingPoint end = correctCoords_(mousePosition);

	if(begin == end)
	{
		addPoint_(begin);
	}
	else
	{
		addLine_(begin, end);
	}

	*drawHelper_ = mousePosition;
	
	if(changed_) repaint(rect());
}

void TFGrayscalePainter::addPoint_(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight_;

	(*view_)[point.x].component1 = yValue;
	(*view_)[point.x].component2 = yValue;
	(*view_)[point.x].component3 = yValue;
	//(*view_)[point.x].alpha = yValue;	//umoznuje prohlizeni i v 3D

	changed_ = true;
}

} // namespace GUI
} // namespace M4D
