#include "TFAbstractPainter.h"

namespace M4D {
namespace GUI {

TFAbstractPainter::TFAbstractPainter():
	painter_(new Ui::TFSimplePainter),
	drawHelper_(NULL),
	margin_(5){

	painter_->setupUi(this);
	paintAreaWidth = width() - 2*margin_;
	paintAreaHeight = height() - 2*margin_;
}

TFAbstractPainter::~TFAbstractPainter(){

	delete painter_;
	if(drawHelper_) delete drawHelper_;
}

void TFAbstractPainter::resize(const QRect rect){

	setGeometry(rect);

	paintAreaWidth = width() - 2*margin_;
	paintAreaHeight = height() - 2*margin_;

	resize_();
}

TFPaintingPoint TFAbstractPainter::correctCoords(const TFPaintingPoint &point){

	return correctCoords(point.x, point.y);
}

TFPaintingPoint TFAbstractPainter::correctCoords(int x, int y){

	int xMax = paintAreaWidth - 1;
	int yMax = paintAreaHeight - 1;
	
	TFPaintingPoint corrected = TFPaintingPoint(x - margin_, margin_ + yMax - y);

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

void TFAbstractPainter::addLine(TFPaintingPoint begin, TFPaintingPoint end){
	
	addLine(begin.x, begin.y, end.x, end.y);
}

void TFAbstractPainter::addLine(int x1, int y1, int x2, int y2){ // assumes x1<x2, |y2-y1|<|x2-x1|
	
	//tfAssert((x1 < x2) && (abs(y2-y1) < abs(x2-x1)));
	if(x1==x2 && y1==y2) addPoint(TFPaintingPoint(x1,y1));

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
			addPoint(TFPaintingPoint(x1,y1));
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
			addPoint(TFPaintingPoint(x1,y1));
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
