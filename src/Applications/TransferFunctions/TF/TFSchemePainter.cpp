
#include "TFSchemePainter.h"
#include "ui_TFSchemePainter.h"

#include <cassert>

TFSchemePainter::TFSchemePainter(int marginH, int marginV): _marginH(marginH), _marginV(marginV), painter(new Ui::TFSchemePainter){

	painter->setupUi(this);
}

TFSchemePainter::~TFSchemePainter(){
	delete painter;
}

void TFSchemePainter::setView(TFScheme* scheme){

	currentView = scheme;
}

void TFSchemePainter::paintEvent(QPaintEvent *){

	QPainter painter(this);
	painter.setPen(Qt::white);

	int painterWidth = FUNCTION_RANGE + 2*_marginH;
	int painterHeight = COLOUR_RANGE + 2*_marginV;
	
	int painterMarginH = (width()- (painterWidth))/2;
	int painterMarginV = (height()-(painterHeight))/2;

	QRect paintingArea(painterMarginH, painterMarginV, painterWidth, painterHeight);
	painter.fillRect(paintingArea, QBrush(Qt::black));

	int beginX = painterMarginH + _marginH;
	int beginY = height() - (painterMarginV + _marginV);
	
	vector<TFSchemePoint*> points = currentView->currentFunction->getAllPoints();
	vector<TFSchemePoint*>::iterator first = points.begin();
	vector<TFSchemePoint*>::iterator end = points.end();
	vector<TFSchemePoint*>::iterator it = first;

	TFSchemePoint origin = TFSchemePoint(beginX, beginY);
	TFSchemePoint* point1 = new TFSchemePoint();

	for(it; it != end; ++it)
	{
		painter.drawLine(origin.x + point1->x, origin.y - point1->y, origin.x + (*it)->x, origin.y - (*it)->y);
		delete point1;
		point1 = *it;
	}

	painter.drawLine(origin.x + point1->x, origin.y - point1->y, beginX + FUNCTION_RANGE, origin.y);
	delete point1;
}

void TFSchemePainter::mousePressEvent(QMouseEvent *e){
	mouseMoveEvent(e);
}

void TFSchemePainter::mouseMoveEvent(QMouseEvent *e){

	TFSchemePoint size = TFSchemePoint(this->width(), this->height());
	TFSchemePoint mousePosition = TFSchemePoint(e->pos().x(), e->pos().y());
	
	int painterMarginH = (size.x - (FUNCTION_RANGE + 2*_marginH)) / 2;
	int painterMarginV = (size.y - (COLOUR_RANGE + 2*_marginV)) / 2;

	int beginX = painterMarginH + _marginH;
	int beginY = size.y - (painterMarginV + _marginV);

	if( mousePosition.x < beginX )
	{
		mousePosition.x = beginX;
	}

	if( mousePosition.x > (beginX + FUNCTION_RANGE) )
	{
		mousePosition.x = beginX + FUNCTION_RANGE;
	}

	if( mousePosition.y > beginY )
	{
		mousePosition.y = beginY;
	}

	if( mousePosition.y < (beginY - COLOUR_RANGE) )
	{
		mousePosition.y = beginY - COLOUR_RANGE;
	}

	assert( (mousePosition.x >= beginX) &&
		(mousePosition.x <= (beginX + FUNCTION_RANGE)) &&
		(mousePosition.y <= beginY) &&
	    (mousePosition.y >= (beginY - COLOUR_RANGE)) );
		
	currentView->currentFunction->addPoint(
		new TFSchemePoint(mousePosition.x - beginX, beginY - mousePosition.y));

	repaint();
}

void TFSchemePainter::Repaint(){
	paintEvent(NULL);
}
