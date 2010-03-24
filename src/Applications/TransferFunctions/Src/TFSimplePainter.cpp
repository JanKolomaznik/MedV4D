
#include "TFSimplePainter.h"
#include "ui_TFSimplePainter.h"

#include <cassert>


TFSimplePainter::TFSimplePainter(): _marginH(10), _marginV(10), _drawHelper(NULL), _painter(new Ui::TFSimplePainter){

	_painter->setupUi(this);
}

TFSimplePainter::TFSimplePainter(int marginH, int marginV): _marginH(marginH), _marginV(marginV), _drawHelper(NULL), _painter(new Ui::TFSimplePainter){

	_painter->setupUi(this);
}

TFSimplePainter::~TFSimplePainter(){

	delete _painter;
	if(_drawHelper) delete _drawHelper;
}

void TFSimplePainter::setup(QWidget *parent){

	setParent(parent);
	show();
}

void TFSimplePainter::setup(QWidget *parent, int marginH, int marginV){

	_marginH = marginH;
	_marginV = marginV;
	setParent(parent);
	show();
}

void TFSimplePainter::resize(const QRect rect){

	setGeometry(rect);
}

void TFSimplePainter::setView(TFPointMap view){

	_view = view;
	repaint(rect());
}

TFPointMap TFSimplePainter::getView(){

	TFPointMap view = _view;
	return view;
}

void TFSimplePainter::paintEvent(QPaintEvent *){

	assert(_view.size() == (width() - 2*_marginH + 1));

	QPainter painter(this);
	painter.setPen(Qt::white);

	painter.fillRect(rect(), QBrush(Qt::black));

	int beginX = _marginH;
	int beginY = height() - _marginV;

	TFPoint origin(beginX, beginY);
	int pointCount = _view.size();
	for(int i = 0; i < pointCount - 1; ++i)
	{
		painter.drawLine(origin.x + i, origin.y - _view[i], origin.x + i + 1, origin.y - _view[i + 1]);
	}
}

void TFSimplePainter::mousePressEvent(QMouseEvent *e){
	//mouseMoveEvent(e);
	_drawHelper = new TFPoint(e->pos().x(), e->pos().y());
}

void TFSimplePainter::mouseReleaseEvent(QMouseEvent *e){

	if(_drawHelper) delete _drawHelper;
	_drawHelper = NULL;
}

void TFSimplePainter::mouseMoveEvent(QMouseEvent *e){

	if(!_drawHelper) return;

	TFPoint mousePosition(e->pos().x(), e->pos().y());
		
	TFPoint begin = painterCoords(*_drawHelper);
	TFPoint end = painterCoords(mousePosition);

	addLine(begin.x, begin.y, end.x, end.y);
	/*
	TFPointMapIterator found = _view.find(newPoint.x);
	if(found == _view.end())
	{
		_view.insert(std::make_pair(newPoint.x, newPoint));
	}
	else
	{
		found->second.y = newPoint.y;
	}*/
	*_drawHelper = mousePosition;
	
	repaint(rect());
}

TFPoint TFSimplePainter::painterCoords(int x, int y){
	return painterCoords(TFPoint(x,y));
}

TFPoint TFSimplePainter::painterCoords(const TFPoint &point){

	int xMax = width() - 2*_marginH;
	int yMax = height() - 2*_marginV;
	TFPoint corrected = TFPoint(point.x - _marginH, _marginV + yMax - point.y);
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

	_view[point.x] = point.y;
	/*
	TFPointMapIterator found = _view.find(point.x);
	if(found == _view.end())
	{
		_view.insert(std::make_pair(point.x, point));
	}
	else
	{
		found->second.y = point.y;
	}
	*/
}

/*
void TFSimplePainter::Repaint(){

	repaint(rect());
}
*/