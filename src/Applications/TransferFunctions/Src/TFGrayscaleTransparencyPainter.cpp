#include "TFGrayscaleTransparencyPainter.h"

namespace M4D {
namespace GUI {

TFGrayscaleTransparencyPainter::TFGrayscaleTransparencyPainter():
	grayChanged_(false),
	transparencyChanged_(false),
	activeView_(ACTIVE_GRAYSCALE){

	grayView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	transparencyView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFGrayscaleTransparencyPainter::~TFGrayscaleTransparencyPainter(){}

void TFGrayscaleTransparencyPainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFGrayscaleTransparencyPainter::setUp(QWidget *parent, int margin){

	margin_ = margin;
	paintAreaWidth = width() - 2*margin_;
	paintAreaHeight = height() - 2*margin_;

	setUp(parent);
}

void TFGrayscaleTransparencyPainter::resize_(){

	grayView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	transparencyView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFFunctionMapPtr TFGrayscaleTransparencyPainter::getGrayscaleView(){

	grayChanged_ = false;
	return grayView_;
}

TFFunctionMapPtr TFGrayscaleTransparencyPainter::getTransparencyView(){

	transparencyChanged_ = false;
	return transparencyView_;
}

bool TFGrayscaleTransparencyPainter::changed(){

	return grayChanged_ || transparencyChanged_;
}

void TFGrayscaleTransparencyPainter::paintEvent(QPaintEvent *){

	QPainter painter(this);
	painter.fillRect(rect(), QBrush(Qt::black));

	int beginX = margin_;
	int beginY = height() - margin_;
	TFPaintingPoint origin(beginX, beginY);

	painter.setPen(Qt::yellow);
	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		painter.drawLine(origin.x + i, origin.y - (*transparencyView_)[i]*paintAreaHeight,
			origin.x + i + 1, origin.y - (*transparencyView_)[i + 1]*paintAreaHeight);
	}

	painter.setPen(Qt::white);
	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		painter.drawLine(origin.x + i, origin.y - (*grayView_)[i]*paintAreaHeight,
			origin.x + i + 1, origin.y - (*grayView_)[i + 1]*paintAreaHeight);
	}
}

void TFGrayscaleTransparencyPainter::mousePressEvent(QMouseEvent *e){

	if(e->button() == Qt::RightButton)
	{
		switch(activeView_)
		{
			case ACTIVE_GRAYSCALE:
			{
				activeView_ = ACTIVE_TRANSPARENCY;
				break;
			}
			case ACTIVE_TRANSPARENCY:
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

void TFGrayscaleTransparencyPainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFGrayscaleTransparencyPainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
		
	TFPaintingPoint begin = correctCoords(*drawHelper_);
	TFPaintingPoint end = correctCoords(mousePosition);

	addLine(begin, end);

	*drawHelper_ = mousePosition;
	
	if(changed()) repaint(rect());
}

void TFGrayscaleTransparencyPainter::addPoint(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight;
	
	switch(activeView_)
	{
		case ACTIVE_GRAYSCALE:
		{
			(*grayView_)[point.x] = yValue;
			grayChanged_ = true;
			break;
		}
		case ACTIVE_TRANSPARENCY:
		{
			(*transparencyView_)[point.x] = yValue;
			transparencyChanged_ = true;
			break;
		}
	}
}

} // namespace GUI
} // namespace M4D
