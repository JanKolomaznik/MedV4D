#include "TFRGBPainter.h"

namespace M4D {
namespace GUI {

TFRGBPainter::TFRGBPainter():
	redChanged_(false),
	greenChanged_(false),
	blueChanged_(false),
	activeView_(ACTIVE_RED){

	redView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	greenView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	blueView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFRGBPainter::~TFRGBPainter(){}

void TFRGBPainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFRGBPainter::setUp(QWidget *parent, int margin){

	margin_ = margin;
	paintAreaWidth = width() - 2*margin_;
	paintAreaHeight = height() - 2*margin_;

	setUp(parent);
}

void TFRGBPainter::resize_(){

	redView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	greenView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	blueView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFFunctionMapPtr TFRGBPainter::getRedView(){

	redChanged_ = false;
	return redView_;
}

TFFunctionMapPtr TFRGBPainter::getGreenView(){

	greenChanged_ = false;
	return greenView_;
}

TFFunctionMapPtr TFRGBPainter::getBlueView(){

	blueChanged_ = false;
	return blueView_;
}

bool TFRGBPainter::changed(){

	return redView_ || greenView_ || blueView_;
}

void TFRGBPainter::paintEvent(QPaintEvent *){

	QPainter painter(this);
	painter.fillRect(rect(), QBrush(Qt::black));

	int beginX = margin_;
	int beginY = height() - margin_;
	TFPaintingPoint origin(beginX, beginY);

	painter.setPen(Qt::red);
	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		painter.drawLine(origin.x + i, origin.y - (*redView_)[i]*paintAreaHeight,
			origin.x + i + 1, origin.y - (*redView_)[i + 1]*paintAreaHeight);
	}

	painter.setPen(Qt::green);
	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		painter.drawLine(origin.x + i, origin.y - (*greenView_)[i]*paintAreaHeight,
			origin.x + i + 1, origin.y - (*greenView_)[i + 1]*paintAreaHeight);
	}

	painter.setPen(Qt::blue);
	for(TFSize i = 0; i < paintAreaWidth - 2; ++i)
	{
		painter.drawLine(origin.x + i, origin.y - (*blueView_)[i]*paintAreaHeight,
			origin.x + i + 1, origin.y - (*blueView_)[i + 1]*paintAreaHeight);
	}
}

void TFRGBPainter::mousePressEvent(QMouseEvent *e){

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
				activeView_ = ACTIVE_RED;
				break;
			}
		}
		return;
	}

	drawHelper_ = new TFPaintingPoint(e->pos().x(), e->pos().y());
	mouseMoveEvent(e);
}

void TFRGBPainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFRGBPainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPaintingPoint mousePosition(e->pos().x(), e->pos().y());
		
	TFPaintingPoint begin = correctCoords(*drawHelper_);
	TFPaintingPoint end = correctCoords(mousePosition);

	addLine(begin, end);

	*drawHelper_ = mousePosition;
	
	if(changed()) repaint(rect());
}

void TFRGBPainter::addPoint(TFPaintingPoint point){

	float yValue = point.y/(float)paintAreaHeight;
	
	switch(activeView_)
	{
		case ACTIVE_RED:
		{
			(*redView_)[point.x] = yValue;
			redChanged_ = true;
			break;
		}
		case ACTIVE_GREEN:
		{
			(*greenView_)[point.x] = yValue;
			greenChanged_ = true;
			break;
		}
		case ACTIVE_BLUE:
		{
			(*blueView_)[point.x] = yValue;
			blueChanged_ = true;
			break;
		}
	}
}

} // namespace GUI
} // namespace M4D
