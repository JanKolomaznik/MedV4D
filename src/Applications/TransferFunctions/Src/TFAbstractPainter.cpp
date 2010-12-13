#include "TFAbstractPainter.h"

namespace M4D {
namespace GUI {

TFAbstractPainter::TFAbstractPainter(QWidget* parent):
	QWidget(parent),
	painter_(new Ui::TFAbstractPainter),
	changed_(true),
	drawHelper_(NULL),
	margin_(5),
	colorBarSize_(10){

	painter_->setupUi(this);
	resize(parent->size());
	show();
}

TFAbstractPainter::~TFAbstractPainter(){

	delete painter_;
	if(drawHelper_) delete drawHelper_;
}

TFColorMapPtr TFAbstractPainter::getView(){

	changed_ = false;
	return view_;
}

bool TFAbstractPainter::changed(){

	return changed_;
}

void TFAbstractPainter::correctView(){

	correctView_();
}

void TFAbstractPainter::paintBackground_(QPainter* drawer, QRect rect, bool hsv){

	QRect paintRect = rect;
	paintRect.setHeight(paintAreaHeight_ + 2*margin_);
	drawer->fillRect(paintRect, QBrush(Qt::black));

	QColor paintingColor;	
	TFSize viewSize = view_->size();
	int xBegin = rect.x() + margin_;
	int yBegin = rect.height();
	for(TFSize i = 0; i < viewSize; ++i)
	{
		if(hsv) paintingColor.setHsvF((*view_)[i].component1, (*view_)[i].component2, (*view_)[i].component3, 1);
		else paintingColor.setRgbF((*view_)[i].component1, (*view_)[i].component2, (*view_)[i].component3, 1);
		drawer->setPen(paintingColor);

		drawer->drawLine(xBegin + i, yBegin, xBegin + i, yBegin - colorBarSize_);
	}

	drawer->setPen(Qt::black);
	drawer->drawLine(xBegin, yBegin - 1, xBegin + viewSize, yBegin - 1);
	drawer->drawLine(xBegin + viewSize, yBegin, xBegin + viewSize, yBegin - colorBarSize_);
	drawer->drawLine(xBegin, yBegin, xBegin, yBegin - colorBarSize_);
	drawer->drawLine(xBegin, yBegin - colorBarSize_, xBegin + viewSize, yBegin - colorBarSize_);
}

TFPaintingPoint TFAbstractPainter::correctCoords_(const TFPaintingPoint &point){

	return correctCoords_(point.x, point.y);
}

TFPaintingPoint TFAbstractPainter::correctCoords_(int x, int y){

	int xMax = paintAreaWidth_ - 1;
	int yMax = paintAreaHeight_;
	
	TFPaintingPoint corrected = TFPaintingPoint(x - margin_, y - margin_);

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

void TFAbstractPainter::addLine_(TFPaintingPoint begin, TFPaintingPoint end){
	
	addLine_(begin.x, begin.y, end.x, end.y);
}

void TFAbstractPainter::addLine_(int x1, int y1, int x2, int y2){ // assumes x1<x2, |y2-y1|<|x2-x1|
	
	//tfAssert((x1 < x2) && (abs(y2-y1) < abs(x2-x1)));
	if(x1==x2 && y1==y2) addPoint_(TFPaintingPoint(x1,y1));

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
			addPoint_(TFPaintingPoint(x1,y1));
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
			addPoint_(TFPaintingPoint(x1,y1));
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

} // namespace GUI
} // namespace M4D
