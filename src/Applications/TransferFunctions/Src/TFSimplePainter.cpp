
#include "TFSimplePainter.h"
#include "ui_TFSimplePainter.h"

#include <cassert>


TFSimplePainter::TFSimplePainter():
	marginH_(10), marginV_(10),
	drawHelper_(NULL),
	painter_(new Ui::TFSimplePainter),
	autoUpdate_(false){

	painter_->setupUi(this);
}

TFSimplePainter::TFSimplePainter(int marginH, int marginV):
	marginH_(marginH), marginV_(marginV),
	drawHelper_(NULL),
	painter_(new Ui::TFSimplePainter),
	autoUpdate_(false){

	painter_->setupUi(this);
}

TFSimplePainter::~TFSimplePainter(){

	delete painter_;
	if(drawHelper_) delete drawHelper_;
}

void TFSimplePainter::setUp(QWidget *parent){

	setParent(parent);
	show();
}

void TFSimplePainter::setUp(QWidget *parent, int marginH, int marginV){

	marginH_ = marginH;
	marginV_ = marginV;
	setParent(parent);
	show();
}

void TFSimplePainter::resize(const QRect rect){

	setGeometry(rect);
}

void TFSimplePainter::setView(TFPointMap view){

	view_ = view;
	repaint(rect());
}

TFPointMap TFSimplePainter::getView(){

	TFPointMap view = view_;
	return view;
}

void TFSimplePainter::setAutoUpdate(bool state){

	autoUpdate_ = state;
}

void TFSimplePainter::paintEvent(QPaintEvent *){

	assert(view_.size() == (width() - 2*marginH_));

	QPainter painter(this);
	painter.setPen(Qt::white);

	painter.fillRect(rect(), QBrush(Qt::black));

	int beginX = marginH_;
	int beginY = height() - marginV_;

	TFPoint origin(beginX, beginY);
	int pointCount = view_.size();
	for(int i = 0; i < pointCount - 2; ++i)
	{
		painter.drawLine(origin.x + i, origin.y - view_[i], origin.x + i + 1, origin.y - view_[i + 1]);
	}
}

void TFSimplePainter::mousePressEvent(QMouseEvent *e){

	drawHelper_ = new TFPoint(e->pos().x(), e->pos().y());
	mouseMoveEvent(e);
}

void TFSimplePainter::mouseReleaseEvent(QMouseEvent *e){

	if(drawHelper_) delete drawHelper_;
	drawHelper_ = NULL;
}

void TFSimplePainter::mouseMoveEvent(QMouseEvent *e){

	if(!drawHelper_) return;

	TFPoint mousePosition(e->pos().x(), e->pos().y());
		
	TFPoint begin = painterCoords(*drawHelper_);
	TFPoint end = painterCoords(mousePosition);

	if(begin == end)
	{
		addPoint(begin);
	}
	else
	{
		addLine(begin.x, begin.y, end.x, end.y);
	}

	*drawHelper_ = mousePosition;
	
	repaint(rect());

	if(autoUpdate_) emit FunctionChanged();
}

TFPoint TFSimplePainter::painterCoords(int x, int y){

	return painterCoords(TFPoint(x,y));
}

TFPoint TFSimplePainter::painterCoords(const TFPoint &point){

	int xMax = width() - 2*marginH_ - 1;
	int yMax = height() - 2*marginV_ - 1;
	
	TFPoint corrected = TFPoint(point.x - marginH_, marginV_ + yMax - point.y);

	if( corrected.x < 0 )
	{
		corrected.x = 0;
	}
	if( corrected.x > xMax )
	{
		corrected.x = xMax;
	}
	if( corrected.y < 0 )
	{
		corrected.y = 0;
	}
	if( corrected.y > yMax )
	{
		corrected.y = yMax;
	}
		
	return corrected;
}

void TFSimplePainter::addLine(int x1, int y1, int x2, int y2){ // assumes x1<x2, |y2-y1|<|x2-x1|
	
    int D, ax, ay, sx, sy;

    sx = x2 - x1;
    ax = abs( sx ) << 1;

    if ( sx < 0 ) sx = -1;
    else if ( sx > 0 ) sx = 1;

    sy = y2 - y1;
    ay = abs( sy ) << 1;

    if ( sy < 0 ) sy = -1;
    else if ( sy > 0 ) sy = 1;

    if ( ax > ay )                          // x coordinate is dominant
    {
		D = ay - (ax >> 1);                   // initial D
		ax = ay - ax;                         // ay = increment0; ax = increment1

		while ( x1 != x2 )
		{
			addPoint(TFPoint(x1,y1));
			if ( D >= 0 )                       // lift up the Y coordinate
			{
				y1 += sy;
				D += ax;
			}
			else
			{
				D += ay;
			}
			x1 += sx;
		}
    }
    else                                    // y coordinate is dominant
    {
		D = ax - (ay >> 1);                   // initial D
		ay = ax - ay;                         // ax = increment0; ay = increment1

		while ( y1 != y2 )
		{
			addPoint(TFPoint(x1,y1));
			if ( D >= 0 )                       // lift up the X coordinate
			{
				x1 += sx;
				D += ay;
			}
			else
			{
				D += ax;
			}
			y1 += sy;
		}
    }
}

void TFSimplePainter::addPoint(TFPoint point){

	view_[point.x] = point.y;
}

void TFSimplePainter::setHistogram(const TFHistogram& histogram){

	//TODO
}

void TFSimplePainter::paintHistogram(const bool paint){

	//TODO
}