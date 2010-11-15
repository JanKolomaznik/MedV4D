#include "TFGrayscalePainter.h"

namespace M4D {
namespace GUI {

TFGrayscalePainter::TFGrayscalePainter(){}

TFGrayscalePainter::~TFGrayscalePainter(){}

void TFGrayscalePainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFGrayscalePainter::setUp(QWidget *parent, int margin){

	setMargin_(margin);
	setUp(parent);
}

void TFGrayscalePainter::paintEvent(QPaintEvent *){

	QPainter painter(this);
	paintBackground_(painter);

	int beginX = margin_;
	int beginY = height() - margin_;
	TFPaintingPoint origin(beginX, beginY);

	painter.setPen(Qt::lightGray);
	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].component1*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].component1*paintAreaHeight);
	}
}

void TFGrayscalePainter::mousePressEvent(QMouseEvent *e){

	drawHelper_ = new TFPaintingPoint(e->pos().x(), e->pos().y());
	mouseMoveEvent(e);
}

void TFGrayscalePainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFGrayscalePainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
		
	TFPaintingPoint begin = correctCoords(*drawHelper_);
	TFPaintingPoint end = correctCoords(mousePosition);

	if(begin == end)
	{
		addPoint(begin);
	}
	else
	{
		addLine(begin, end);
	}

	*drawHelper_ = mousePosition;
	
	if(changed_) repaint(rect());
}

void TFGrayscalePainter::addPoint(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight;

	(*view_)[point.x].component1 = yValue;
	(*view_)[point.x].component2 = yValue;
	(*view_)[point.x].component3 = yValue;
	//(*view_)[point.x].alpha = yValue;	//umoznuje prohlizeni i v 3D

	changed_ = true;
}

} // namespace GUI
} // namespace M4D
