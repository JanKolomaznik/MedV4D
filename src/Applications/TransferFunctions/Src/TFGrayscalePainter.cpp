#include "TFGrayscalePainter.h"

namespace M4D {
namespace GUI {

TFGrayscalePainter::TFGrayscalePainter():
	changed_(false){

	view_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFGrayscalePainter::~TFGrayscalePainter(){}

void TFGrayscalePainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFGrayscalePainter::setUp(QWidget *parent, int margin){

	margin_ = margin;
	paintAreaWidth = width() - 2*margin_;
	paintAreaHeight = height() - 2*margin_;

	setUp(parent);
}

void TFGrayscalePainter::resize_(){

	view_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFFunctionMapPtr TFGrayscalePainter::getView(){

	changed_ = false;
	return view_;
}

bool TFGrayscalePainter::changed(){

	return changed_;
}

void TFGrayscalePainter::paintEvent(QPaintEvent *){

	QPainter painter(this);
	painter.setPen(Qt::white);

	painter.fillRect(rect(), QBrush(Qt::black));

	int beginX = margin_;
	int beginY = height() - margin_;

	TFPaintingPoint origin(beginX, beginY);
	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		painter.drawLine(origin.x + i, origin.y - (*view_)[i]*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i + 1]*paintAreaHeight);
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
	(*view_)[point.x] = yValue;
	changed_ = true;
}

} // namespace GUI
} // namespace M4D
