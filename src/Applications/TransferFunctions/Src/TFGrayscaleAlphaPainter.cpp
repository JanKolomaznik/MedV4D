#include "TFGrayscaleAlphaPainter.h"

namespace M4D {
namespace GUI {

TFGrayscaleAlphaPainter::TFGrayscaleAlphaPainter():
	activeView_(ACTIVE_GRAYSCALE){}

TFGrayscaleAlphaPainter::~TFGrayscaleAlphaPainter(){}

void TFGrayscaleAlphaPainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFGrayscaleAlphaPainter::setUp(QWidget *parent, int margin){

	setMargin_(margin);
	setUp(parent);
}

void TFGrayscaleAlphaPainter::paintEvent(QPaintEvent *){

	QPainter painter(this);
	paintBackground_(painter);

	int beginX = margin_;
	int beginY = height() - margin_;
	TFPaintingPoint origin(beginX, beginY);

	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		//alpha
		painter.setPen(Qt::yellow);
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].alpha*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].alpha*paintAreaHeight);
		//gray
		painter.setPen(Qt::lightGray);
		painter.drawLine(origin.x + i, origin.y - (*view_)[i].component1*paintAreaHeight,
			origin.x + i + 1, origin.y - (*view_)[i+1].component1*paintAreaHeight);
	}
}

void TFGrayscaleAlphaPainter::mousePressEvent(QMouseEvent *e){

	if(e->button() == Qt::RightButton)
	{
		switch(activeView_)
		{
			case ACTIVE_GRAYSCALE:
			{
				activeView_ = ACTIVE_ALPHA;
				break;
			}
			case ACTIVE_ALPHA:
			{
				activeView_ = ACTIVE_GRAYSCALE;
				break;
			}
		}
		return;
	}

	drawHelper_ = new TFPaintingPoint(e->pos().x(), e->pos().y());
	mouseMoveEvent(e);
}

void TFGrayscaleAlphaPainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFGrayscaleAlphaPainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
		
	TFPaintingPoint begin = correctCoords(*drawHelper_);
	TFPaintingPoint end = correctCoords(mousePosition);

	addLine(begin, end);

	*drawHelper_ = mousePosition;
	
	if(changed()) repaint(rect());
}

void TFGrayscaleAlphaPainter::addPoint(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight;
	
	switch(activeView_)
	{
		case ACTIVE_GRAYSCALE:
		{
			(*view_)[point.x].component1 = yValue;
			(*view_)[point.x].component2 = yValue;
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
