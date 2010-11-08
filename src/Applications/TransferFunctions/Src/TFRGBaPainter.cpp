#include "TFRGBaPainter.h"

namespace M4D {
namespace GUI {

TFRGBaPainter::TFRGBaPainter():
	redChanged_(false),
	greenChanged_(false),
	blueChanged_(false),
	transparencyChanged_(false),
	activeView_(ACTIVE_RED){

	redView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	greenView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	blueView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	transparencyView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFRGBaPainter::~TFRGBaPainter(){}

void TFRGBaPainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFRGBaPainter::setUp(QWidget *parent, int margin){

	margin_ = margin;
	paintAreaWidth = width() - 2*margin_;
	paintAreaHeight = height() - 2*margin_;

	setUp(parent);
}

void TFRGBaPainter::resize_(){

	redView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	greenView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	blueView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
	transparencyView_ = TFFunctionMapPtr(new TFFunctionMap(paintAreaWidth));
}

TFFunctionMapPtr TFRGBaPainter::getRedView(){

	redChanged_ = false;
	return redView_;
}

TFFunctionMapPtr TFRGBaPainter::getGreenView(){

	greenChanged_ = false;
	return greenView_;
}

TFFunctionMapPtr TFRGBaPainter::getBlueView(){

	blueChanged_ = false;
	return blueView_;
}
TFFunctionMapPtr TFRGBaPainter::getTransparencyView(){

	transparencyChanged_ = false;
	return transparencyView_;
}

bool TFRGBaPainter::changed(){

	return redView_ || greenView_ || blueView_ || transparencyChanged_;
}

void TFRGBaPainter::paintEvent(QPaintEvent *){

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
				activeView_ = ACTIVE_TRANSPARENCY;
				break;
			}
			case ACTIVE_TRANSPARENCY:
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
