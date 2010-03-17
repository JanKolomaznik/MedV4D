
#include "TFSimplePainter.h"
#include "ui_TFSimplePainter.h"

#include <cassert>


TFSimplePainter::TFSimplePainter(): _marginH(10), _marginV(10), _painter(new Ui::TFSimplePainter){

	_painter->setupUi(this);
}

TFSimplePainter::TFSimplePainter(int marginH, int marginV): _marginH(marginH), _marginV(marginV), _painter(new Ui::TFSimplePainter){

	_painter->setupUi(this);
}

TFSimplePainter::~TFSimplePainter(){
	delete _painter;
}

void TFSimplePainter::setup(QWidget *parent, const QRect rect){

	setGeometry(rect);
	setParent(parent);
	show();
}

void TFSimplePainter::setup(QWidget *parent, const QRect rect, int marginH, int marginV){

	_marginH = marginH;
	_marginV = marginV;
	setGeometry(rect);
	setParent(parent);
	show();
}

void TFSimplePainter::setView(TFPointMap view){

	_view = view;
	emit Repaint();
}

TFPointMap TFSimplePainter::getView(){

	TFPointMap view = _view;
	return view;
}

void TFSimplePainter::paintEvent(QPaintEvent *){

	QPainter painter(this);
	painter.setPen(Qt::white);

	int painterWidth = FUNCTION_RANGE_SIMPLE + 2*_marginH;
	int painterHeight = COLOR_RANGE_SIMPLE + 2*_marginV;
	
	int painterMarginH = (width()- (painterWidth))/2;
	int painterMarginV = (height()-(painterHeight))/2;

	QRect paintingArea(painterMarginH, painterMarginV, painterWidth, painterHeight);
	painter.fillRect(paintingArea, QBrush(Qt::black));

	int beginX = painterMarginH + _marginH;
	int beginY = height() - (painterMarginV + _marginV);
	
	TFPointMapIterator first = _view.begin();
	TFPointMapIterator end = _view.end();
	TFPointMapIterator it = first;

	TFPoint origin(beginX, beginY);
	TFPoint point1;

	for(it; it != end; ++it)
	{
		painter.drawLine(origin.x + point1.x, origin.y - point1.y, origin.x + it->second.x, origin.y - it->second.y);
		point1 = it->second;
	}

	painter.drawLine(origin.x + point1.x, origin.y - point1.y, beginX + FUNCTION_RANGE_SIMPLE, origin.y);
}

void TFSimplePainter::mousePressEvent(QMouseEvent *e){
	mouseMoveEvent(e);
}

void TFSimplePainter::mouseMoveEvent(QMouseEvent *e){

	TFPoint size(this->width(), this->height());
	TFPoint mousePosition(e->pos().x(), e->pos().y());
	
	int painterMarginH = (size.x - (FUNCTION_RANGE_SIMPLE + 2*_marginH)) / 2;
	int painterMarginV = (size.y - (COLOR_RANGE_SIMPLE + 2*_marginV)) / 2;

	int beginX = painterMarginH + _marginH;
	int beginY = size.y - (painterMarginV + _marginV);

	if( mousePosition.x < beginX )
	{
		mousePosition.x = beginX;
	}

	if( mousePosition.x > (beginX + FUNCTION_RANGE_SIMPLE) )
	{
		mousePosition.x = beginX + FUNCTION_RANGE_SIMPLE;
	}

	if( mousePosition.y > beginY )
	{
		mousePosition.y = beginY;
	}

	if( mousePosition.y < (beginY - COLOR_RANGE_SIMPLE) )
	{
		mousePosition.y = beginY - COLOR_RANGE_SIMPLE;
	}

	assert( (mousePosition.x >= beginX) &&
		(mousePosition.x <= (beginX + FUNCTION_RANGE_SIMPLE)) &&
		(mousePosition.y <= beginY) &&
	    (mousePosition.y >= (beginY - COLOR_RANGE_SIMPLE)) );
		
	TFPoint newPoint(mousePosition.x - beginX, beginY - mousePosition.y);

	TFPointMapIterator found = _view.find(newPoint.x);

	if(found == _view.end())
	{
		_view.insert(std::make_pair(newPoint.x, newPoint));
	}
	else
	{
		found->second.y = newPoint.y;
	}

	repaint(rect());
}

void TFSimplePainter::Repaint(){

	repaint(rect());
}