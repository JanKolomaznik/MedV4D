#include "TFSimplePainter.h"

namespace M4D {
namespace GUI {

TFSimplePainter::TFSimplePainter():
	changed_(false){

	view_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFSimplePainter::~TFSimplePainter(){}

void TFSimplePainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFSimplePainter::setUp(QWidget *parent, int margin){

	margin_ = margin;
	paintAreaWidth = width() - 2*margin_;
	paintAreaHeight = height() - 2*margin_;

	setUp(parent);
}

void TFSimplePainter::resize_(){

	view_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFFunctionMapPtr TFSimplePainter::getView(){

	changed_ = false;
	return view_;
}

bool TFSimplePainter::changed(){

	return changed_;
}

void TFSimplePainter::paintEvent(QPaintEvent *){

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

void TFSimplePainter::mousePressEvent(QMouseEvent *e){

	drawHelper_ = new TFPaintingPoint(e->pos().x(), e->pos().y());
	mouseMoveEvent(e);
}

void TFSimplePainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFSimplePainter::mouseMoveEvent(QMouseEvent *e){

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

void TFSimplePainter::addPoint(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight;
	(*view_)[point.x] = yValue;
	changed_ = true;
}

} // namespace GUI
} // namespace M4D
